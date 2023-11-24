import os
import gc
import csv
import numpy as np
import torch
import pickle

from tqdm import tqdm
import torch.nn.functional as F
import mmcv
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.image import tensor2imgs
from mmdet.core import encode_mask_results
from mmcv.parallel import collate, scatter, MMDataParallel, MMDistributedDataParallel
from mmdet.datasets.pipelines import Compose
from mmdet.datasets import CocoDataset
from mmcv.runner import (
    get_dist_info,
    init_dist,
    load_checkpoint,
    wrap_fp16_model,
)
from mmdet.apis import init_detector, inference_detector, show_result_pyplot, multi_gpu_test, single_gpu_test
from mmdet.datasets import (
    build_dataloader,
    build_dataset,
    replace_ImageToTensor,
)
from mmdet.models import build_detector

# from id_to_class import bdd100k_det as id2class
from id_to_class import shift_det as id2class

import torch.multiprocessing

sharing_strategy = "file_system"
torch.multiprocessing.set_sharing_strategy(sharing_strategy)

def set_worker_sharing_strategy(worker_id: int) -> None:
    torch.multiprocessing.set_sharing_strategy(sharing_strategy)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")



def cls_to_probs(cls_scores, gt_bboxes):

    probs = F.softmax(cls_scores, dim=-1).cpu().numpy()
    assert cls_scores.size(0) == gt_bboxes.size(0)

    # sanity check
    # close_to_1 = probs.sum(axis=1)
    # print(f'close_to_1={close_to_1}')

    return probs


def cal_loss(probs, gt_labels):
    log_probs = np.log(probs[:,:-1])
    log_probs_tensor = torch.Tensor(log_probs)
    gt_labels_tensor = torch.LongTensor(gt_labels)

    # F.nll_loss basically do nothing but flip the sign of the input
    losses = F.nll_loss(log_probs_tensor, gt_labels_tensor, reduction='none').cpu().numpy()
    return losses

def gpu_test(model,
             data_loader,
             csv_path,
             show=False,
             out_dir=None,
             show_score_thr=0.3):
    
    model.eval()
    results = []
    dataset = data_loader.dataset
    PALETTE = getattr(dataset, 'PALETTE', None)
    prog_bar = mmcv.ProgressBar(len(dataset))

    saved_objs = []

    print(f'Creating csv file: {csv_path}')
    with open(f'{csv_path}', 'w', newline='') as f:
        cnt = 0
        writer = csv.writer(f)

        # BDD100k (without true labels)
        # header_line = ',imgpath,cls_score,prob,max_prob,bbox_gt,label_gt,label_gt_des,label_pred,label_pred_des,loss\n'

        # SHIFT (with true labels)
        header_line = ',imgpath,cls_score,prob,max_prob,bbox_gt,true_label,true_label_des,label_gt,label_gt_des,label_pred,label_pred_des,loss\n'
        f.write(header_line)

        for i, data in enumerate(tqdm(data_loader)):
            

            assert len(data["img"]) == 1

            if data == None:
                continue

            # BDD100K and SHIFT
            metas = data["img_metas"].data[0][0]

            gt_bboxes = data['gt_bboxes'][0].squeeze(0).detach().cpu()
            gt_labels = data['gt_labels'][0].squeeze(0).detach().cpu()
            truelabels = data['truelabels'][0].squeeze(0).detach().cpu().tolist()

            data["img_metas"] = [data["img_metas"]]

            with torch.no_grad():
                #############################################################
                #     Directly extract features from the backbone+neck.     #
                #############################################################
                # feature = model.extract_feats(data['img'])
                
                # results = model.show_result(data['img'])
                # result = model(return_loss=False, rescale=True, **data)

                #############################################################
                #        Predicted Probabilities for single image           #
                #############################################################
                proposals = data["gt_bboxes"][0][0]

                model = model.to(device)
                proposals = proposals.to(device)

                out_dummy = model.forward_inject_gt(data['img'][0].to(device), proposals)
                
                data['img'][0] = data['img'][0].detach().cpu()
                proposals = proposals.detach().cpu()

                cls_score = out_dummy["cls_score"].detach().cpu()
                cls_score = cls_score.squeeze(0)

                pred_bbox = out_dummy["pred_bbox"].detach().cpu()
                assert torch.is_tensor(cls_score)
                assert torch.is_tensor(pred_bbox)

                # SHIFT
                if cls_score.size(0) != gt_bboxes.size(0): 
                    cls_score=cls_score.unsqueeze(0)

                num_bbox = cls_score.size(0)

                # probs included bg! but you can re-calculate it from cls_scores
                probs = cls_to_probs(cls_score, gt_bboxes)

                # background bg prob is pruned out in cal_loss
                loss = cal_loss(probs, gt_labels)
                max_prob_idx = np.argmax(probs, axis=1)
                max_prob = np.max(probs, axis=1)

                for i in range(num_bbox):
                    label_gt_des = id2class[gt_labels[i].detach().cpu().item()]
                    label_pred_des = id2class[max_prob_idx[i]]
                    true_label = truelabels[i]
                    true_label_des = id2class[truelabels[i]]
                    prob_str = ';'.join([f'{i:.17f}' for i in probs[i]])
                    cls_score_str = ';'.join([f'{i:.17f}' for i in cls_score[i].detach().cpu().tolist()])
                    gt_bboxes_str = ';'.join([f'{int(i)}' for i in gt_bboxes[i].detach().cpu().tolist()])
                    

                    row = [
                        cnt,
                        metas['filename'][0], # imgpath
                        cls_score_str,
                        prob_str,
                        max_prob[i],
                        gt_bboxes_str,
                        true_label,
                        true_label_des,
                        gt_labels[i].detach().cpu().item(),
                        label_gt_des,
                        max_prob_idx[i], # label_pred
                        label_pred_des,
                        loss[i],
                    
                    ]
                    writer.writerow(row)
                    cnt+=1

                data = None
                out_dummy = None
                del out_dummy
                del cls_score
                del pred_bbox
                del proposals
                del data
                del metas
                gc.collect()
                torch.cuda.empty_cache()




def main(m_name, conf_name, target_evaluation):

    cwd = os.getcwd()

    ######################### BDD100K #########################
    # # m_name = 'faster_rcnn_r50_fpn_1x_det_bdd100K_TrainFromScratch'
    # # m_name = 'faster_rcnn_swin-t_fpn_1x_det_bdd100k_PreSwinTiny'
    # conf_path = f'/media/18T/model_thesis/{m_name}/{conf_name}'
    # model_dirpath = f'/media/18T/model_thesis/{m_name}'


    # csv_dirpath = f'{eval_dirpath}/yu/data_pred/bdd100k_fasterrcnn/{target_evaluation}'

    
    ######################### SHIFT #########################
    parent_dir = f'{data_dirpath}/SHIFT'
    code_dir = cwd
    conf_path = f'{code_dir}/configs/_base_/models/faster_rcnn_r50_fpn_1x_det_SHIFT.py'

    epo = 1
    model_path = f'{cwd}/work_dirs/faster_rcnn_r50_fpn_1x_det_SHIFT_uni10/epoch_{epo}.pth'


    model_dirpath = f'{m_name}'
    conf_path = f'{model_dirpath}/faster_rcnn_r50_fpn_1x_det_SHIFT.py'

    # Choose the target csv dirpath
    target_csv_dirpath = ''
    target_csv_filename = target_evaluation

    csv_dirpath = f'{target_csv_dirpath}/{target_csv_filename}'

    # example: csv_dirpath '
    # csv_dirpath = f'./trafficsign/yu/data_pred/SHIFT/{target_csv_filename}'

    if not os.path.isdir(csv_dirpath):
        print(f'Create csv_dirpath: {csv_dirpath}')
        os.mkdir(csv_dirpath)

    epos = range(1, 9)

    for epo in epos:
        # BDD100k on FasterRCNN
        # model_path = f'{model_dirpath}/epoch_{epo}.pth'

        # SHIFT on FasterRCNN
        model_path = f'{model_dirpath}/epoch_{epo}.pth'

        csv_path = f'{csv_dirpath}/epo{epo}.csv'

        cfg = Config.fromfile(conf_path)

        # bdd100k on FasteRCNN
        if target_evaluation in ['swin-t_val_split0', 'val_split0']:
            input_val = cfg.data.val_split0
        elif target_evaluation in ['swin-t_val_split1', 'val_split1']:
            input_val = cfg.data.val_split1
        elif target_evaluation in ['swin-t_val_all', 'val_all']:
            input_val = cfg.data.val_all
            
        elif target_evaluation[:6] == 'train_':
            input_val = cfg.data.val_train
        else:
            input_val = cfg.data.val


        cfg.load_from = model_path

        # set cudnn_benchmark
        if cfg.get("cudnn_benchmark", False):
            torch.backends.cudnn.benchmark = True

        cfg.model.pretrained = None
        if cfg.model.get("neck"):
            if isinstance(cfg.model.neck, list):
                for neck_cfg in cfg.model.neck:
                    if neck_cfg.get("rfp_backbone"):
                        if neck_cfg.rfp_backbone.get("pretrained"):
                            neck_cfg.rfp_backbone.pretrained = None
            elif cfg.model.neck.get("rfp_backbone"):
                if cfg.model.neck.rfp_backbone.get("pretrained"):
                    cfg.model.neck.rfp_backbone.pretrained = None

        # in case the test dataset is concatenated
        samples_per_gpu = 1

        ######################### BDD100K #########################
        # cfg.data.test = cfg.data.offical_val_split
        ######################### SHIFT #########################
        cfg.data.test = input_val
        if isinstance(cfg.data.test, dict):
            cfg.data.test.test_mode = True  # type: ignore
            samples_per_gpu = cfg.data.test.pop("samples_per_gpu", 1)
            if samples_per_gpu > 1:
                # Replace 'ImageToTensor' to 'DefaultFormatBundle'
                cfg.data.test.pipeline = replace_ImageToTensor(  # type: ignore
                    cfg.data.test.pipeline  # type: ignore
                )
        elif isinstance(cfg.data.test, list):
            for ds_cfg in cfg.data.test:
                ds_cfg.test_mode = True
            samples_per_gpu = max(
                [ds_cfg.pop("samples_per_gpu", 1) for ds_cfg in cfg.data.test]
            )
            if samples_per_gpu > 1:
                for ds_cfg in cfg.data.test:
                    ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

        rank, _ = get_dist_info()

        # build the dataloader
        distributed = False
        dataset = build_dataset(cfg.data.test)
        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False
        )
    

        # build the model and load checkpoint
        cfg.model.train_cfg = None

        model = init_detector(cfg, model_path, device=device)

        ######################### BDD100K #########################
        # model.CLASSES = [
        #     'pedestrian',
        #     'rider',
        #     'car',
        #     'truck',
        #     'bus',
        #     'train',
        #     'motorcycle',
        #     'bicycle',
        #     'traffic light',
        #     'traffic sign'
        # ]
        ######################### SHIFT #########################
        model.CLASSES = [
            "pedestrian",
            "car", 
            "truck", 
            "bus", 
            "motorcycle", 
            "bicycle"
        ]

        fp16_cfg = cfg.get("fp16", None)

        if fp16_cfg is not None:
            wrap_fp16_model(model)

        checkpoint = load_checkpoint(model, cfg.load_from, map_location="cpu")

        fuse_conv_bn = False

        if fuse_conv_bn:
            model = fuse_conv_bn(model)

        # old versions did not save class info in checkpoints, this walkaround is
        # for backward compatibility
        if "CLASSES" in checkpoint.get("meta", {}):
            model.CLASSES = checkpoint["meta"]["CLASSES"]
        else:
            model.CLASSES = dataset.CLASSES

        gpu_test(model, data_loader, csv_path)



if __name__ == '__main__':
    import sys
    main(sys.argv[1], sys.argv[2], sys.argv[3])
