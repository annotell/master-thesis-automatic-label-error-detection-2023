from argparse import ArgumentParser
from matplotlib import pyplot as plt
import torch
import numpy as np
from mmcv.parallel import collate, scatter

from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from mmdet.datasets.pipelines import Compose
from torchsummary import summary
from mmdet.datasets import (
    build_dataloader,
    build_dataset,
    replace_ImageToTensor,
)

device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")



def main():
    parser = ArgumentParser()
    parser.add_argument('--img', help='Image file')
    parser.add_argument('--config', help='Config file')
    parser.add_argument('--checkpoint', help='Checkpoint file')
    args = parser.parse_args()

    parent_dir = f'{data_dirpath}/bdd100k/bdd100k_images_100k'
    code_dir = '/home/belay/Documents/0_UPPSALA/KOGNIC_research/Codes/Swin-Transformer-Object-Detection'
    label_path = f'{parent_dir}/labels_cofrom mmdet.datasets.pipelines import Composeo2/val_cocofmt.json'
    img_dir = f'{parent_dir}/images/100k/val/'

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device='cuda:0')

    # coco
    # {1: 'person', 2: 'car', 3: 'rider', 4: 'bus', 5: 'truck', 6: 'bike', 7: 'motor', 8: 'traffic light', 9: 'traffic sign'}

    # Official
    # 1: pedestrian    parent_dir = f'{data_dirpath}/bdd100k/bdd100k_images_100k'
    code_dir = '/home/belay/Documents/0_UPPSALA/KOGNIC_research/Codes/Swin-Transformer-Object-Detection'
    label_path = f'{parent_dir}/labels_cofrom mmdet.datasets.pipelines import Composeo2/val_cocofmt.json'
    img_dir = f'{parent_dir}/images/100k/val/'
    # 2: rider
    # 3: car
    # 4: truck
    # 5: bus
    # 6: train
    # 7: motorcycle
    # 8: bicycle
    # 9: traffic light
    # 10: traffic sign

    # COCO
    # model.CLASSES = [
    #     'person',
    #     'car',
    #     'rider',
    #     'bus',
    #     'truck',
    #     'bike',
    #     'motor',
    #     'traffic light',
    #     'traffic sign'
    # ]

    # Test
    # model.CLASSES = [
    #     'car',
    #     'person',
    #     'rider',
    #     'truck',
    #     'traffic light',
    #     'traffic sign'
    #     'bus',
    #     'motor',
    #     'bike',
    # ]

    # Officialpython embedding.py --img /media/18T/data_thesis/bdd100k/bdd100k_images_100k/0/val/02521568-dd1da95c.jpg --config configs/_base_/models/faster_rcnn_r50_fpn_1x_det_bdd100k.py --checkpoint work_dirs/faster_rcnn_r50_fpn_1x_det_bdd100k_fold0/
    model.CLASSES = [
        'pedestrian',
        'rider',
        'car',
        'truck',
        'bus',
        'train',
        'motorcycle',
        'bicycle',
        'traffic light',
        'traffic sign'
    ]
    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    # build the dataloader
    distributed = False
    samples_per_gpu = 1
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
    )

    # prepare data
    data = dict(img_info=dict(filename=args.img), img_prefix=None)
    # build the data pipeline
    test_pipeline = Compose(cfg.data.test.pipeline)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    data = scatter(data, [device])[0]
    # print(dir(data))
    # return

    img = data['img'][0]
    # x = model.extract_feat(img)
    
    # result, result_all = inference_detector(model, args.img)
    # print(f'len of result={len(result)}')
    # print(f'0 = {result[0].shape}')
    # print(f'1 = {result[1].shape}')
    # print(f'2 = {result[2].shape}, {result[2][0]}, , {result[2][1]}')
    # print(f'3 = {result[3].shape}')
    # print(f'4 = {result[4].shape}')
    # print(f'5 = {result[5].shape}')
    # print(f'6 = {result[6].shape}')
    # print(f'7 = {result[7].shape}')
    # print('-------------------------')
    # show_result_pyplot(model, args.img, result)

    proposals = torch.randn(1000, 4).to(img.device)
    out_dummy = model.forward_inject_gt(img, proposals)
    # rpn_outs, roi_outs = out_dummy
    # print(f'out_dummy({out_dummy[0][0].shape})')
    # print(f'rpn_outs({out_dummy["rpn_outs"][0][0].shape})={out_dummy["rpn_outs"][0][0]}')
    # print(f'cls_score({out_dummy["cls_score"].shape})={out_dummy["cls_score"]}')
    # print(f'pred_bbox({out_dummy["pred_bbox"].shape})={out_dummy["pred_bbox"]}')

    # feat = np.squeeze(feat, axis=0)
    # # .detach().cpu().numpy()

    # for fe in range(len(feat)):
    #     plt.imshow(feat[fe])
    #     plt.show()
    
    # plt.imshow(feat[1])
    # plt.show()

    # plt.imshow(feat[2])
    # plt.show()

    # # hack your favorite box here, first 0 represents batch indices
    # # you can just set it as 0
    # rois = torch.tensor([
    #     [0., 0., 10., 5., 20.],
    #     [0., 10., 20., 15., 24.3],
    # ]).to(device)

    # print(img.shape)
    
    # # print(summary(model, (1, 3, 736, 1280), device='cuda'))
    # # return
    # bbox_feats = model.roi_head.bbox_roi_extractor(
    #     x[:model.roi_head.bbox_roi_extractor.num_inputs], rois)
    # print(bbox_feats.shape)


if __name__ == '__main__':
    main()