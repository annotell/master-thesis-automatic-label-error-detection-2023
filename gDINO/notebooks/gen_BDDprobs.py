import os
import gc
import csv
import json
import pickle
import pprint
import traceback
import numpy as np
import matplotlib.pyplot as plt
import supervision as sv
from tqdm import tqdm
import torch.nn.functional as F
from tqdm import tqdm

from id_to_class import bdd100k_det as id2class
from groundingdino.util.inference import load_model, load_image, load_copped_image, predict, annotate


class2id = {v: k for k, v in id2class.items()}

pp = pprint.PrettyPrinter(indent=4)


def xywh2ratio(img_w, img_h, bbox):

    x, y, w, h = bbox
    x_center = x + w/2.0
    y_center = y + h/2.0
    x_center_ratio = x_center/img_w * 1.0
    y_center_ratio = y_center/img_h * 1.0
    w_ratio = w/img_w * 1.0
    h_ratio = h/img_h * 1.0


    return [x_center_ratio, y_center_ratio, w_ratio, h_ratio]


bdd100k_all = {
    0: "pedestrian",
    1: "rider",
    2: "car",
    3: "truck",
    4: "bus",
    5: "train",
    6: "motorcycle",
    7: "bicycle",
    8: "traffic light",
    9: "traffic sign"
}

bdd100k_all_map = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9
}

# PROMPT TRAFFIC
bdd100k_traffic = {
    0: "traffic light",
    1: "traffic sign"
}
bdd100k_traffic_map = {
    8: 0,
    9: 1
}

# PROMPT VEHICLE
bdd100k_vehicle = {
    0: "car",
    1: "truck",
    2: "bus",
    3: "train"
}
bdd100k_vehicle_map = {
    2: 0,
    3: 1,
    4: 2,
    5: 3,
}

# PROMPT PEOPLE
bdd100k_people = {
    0: "pedestrian",
    1: "rider",
    2: "motorcycle",
    3: "bicycle"
}

bdd100k_people_properties = {
    4: "man",
    5: "woman",
    6: "child"
}

bdd100k_people_map = {
    0: 0,
    1: 1,
    6: 2,
    7: 3,
}

subset_name = 'all'
det_subset = bdd100k_all
det_map = bdd100k_all_map

# subset_name = 'traffic'
# det_subset = bdd100k_traffic
# det_map = bdd100k_traffic_map

# subset_name = 'vehicle'
# det_subset = bdd100k_vehicle
# det_map = bdd100k_vehicle_map

# subset_name = 'people'
# det_subset = bdd100k_people
# det_map = bdd100k_people_map


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


def logits_to_probs(logits):

    probs = F.softmax(logits, dim=-1).cpu().numpy()

    # sanity check
    #close_to_1 = probs.sum(axis=1)
    #print(f'close_to_1={close_to_1}')

    return probs


def main(json_name):

    username = os.getlogin()
    home = f'/home/{username}/Documents/github_others/gDINO'
    pp = pprint.PrettyPrinter(indent=4)
    print(f'HOME={home}')

    # weight_name_acronym ='cog'
    # weight_name = 'groundingdino_swinb_cogcoor.pth'
    # conf_path = f'{home}/groundingdino/config/GroundingDINO_SwinB.cfg.py'

    weight_name_acronym ='ogc'
    weight_name = "groundingdino_swint_ogc.pth"
    conf_path = f'{home}/groundingdino/config/GroundingDINO_SwinT_OGC.py'

    weight_path = os.path.join(home, "notebooks/weights", weight_name)


    print(weight_path, "; exist:", os.path.isfile(weight_path))
    print(conf_path, "; exist:", os.path.isfile(conf_path))

    
    model = load_model(conf_path, weight_path)

    BOX_TRESHOLD = 0.2
    TEXT_TRESHOLD = 0.15

    ############################################
    #                 BDD100K                  #
    ############################################
    # you need to set up the path to the dataset
    data_dirpath = ''
    target_csv_dirpath = ''
    json_dirname = 'labels_coco2'
    # json_name = 'val_cocofmt'

    if json_name == 'val_cocofmt':
        image_dir = f'{data_dirpath}/bdd100k_images_100k/images/100k/val'
        
    elif json_name == 'train_cocofmt':
        image_dir = f'{data_dirpath}/bdd100k_images_100k/images/100k/train'

    json_path = f'{data_dirpath}/bdd100k_images_100k/{json_dirname}/{json_name}.json'

    if weight_name_acronym == 'cog':
        target_csv_dirpath = f'{target_csv_dirpath}/yu/data_pred/bdd100k_gdino_cog_test'

    elif weight_name_acronym == 'ogc':
        target_csv_dirpath = f'{target_csv_dirpath}/yu/data_pred/bdd100k_gdino_ogc_test'

    target_csv_path = f'{target_csv_dirpath}/{json_name}_box{BOX_TRESHOLD:.1f}_{subset_name}.csv'

    with open(json_path, 'r') as f:
        data_dict = json.load(f)

    def create_idx_mapping(oriset, subset):
        new_classnames = list(subset.values())
        ori2subset = {}
        for k, v in oriset.items():
            if v in new_classnames:
                new_idx = new_classnames.index(v)
                ori2subset[k] = new_idx
        return ori2subset


    print(f'\n========== Process {len(data_dict["images"])} images ==========')
    print(f'\n========== Process {list(data_dict.keys())} images ==========')

    print(f'Creating csv file: {target_csv_path}')

    if os.path.isfile(target_csv_path):
        print(f'File existed! {target_csv_path}')
        print(f'You will overwrite the file!')
    

    with open(f'{target_csv_path}', 'w', newline='') as f:
        cnt = 0
        writer = csv.writer(f)
        header_line = ',imgpath,bbox_gt,bbox_pred,label_gt,label_gt_des,label_pred,label_pred_des,class_logit,class_logit_raw,prob,max_prob,prob_sig,max_prob_sig\n'
        f.write(header_line)

       
        # for frame in tqdm(data_dict['frames']):
        for idx, coco_img_obj in tqdm(enumerate(data_dict['images'])):

            imgname = coco_img_obj['file_name']
            imgpath = f"{image_dir}/{imgname}"
            img_shape = (coco_img_obj['width'], coco_img_obj['height'])

            assert os.path.isfile(imgpath), f'Not existed! {imgpath}'

            gt_bboxes = []
            labels_gt = []
            labels_gt_des = []
            for anno in data_dict['annotations']:


                if coco_img_obj['id'] == anno['image_id']:

                    bbox = xywh2ratio(coco_img_obj['width'], coco_img_obj['height'], anno['bbox'])

                    # bdd100k
                    classid = anno["category_id"]-1
                    # id2class is the original classes for the json file
                    classname = id2class[classid] # use all to map the original coco format

                    if classname not in list(det_subset.values()):
                        continue

                    new_classid = det_map[classid]
                    
                    gt_bboxes.append(bbox)
                    labels_gt.append(new_classid)
                    labels_gt_des.append(classname)

            try:
                if len(gt_bboxes) == 0:
                    # print(f'*** Skip empty frame! ***')
                    # print(f'cnt={cnt}, imgpath={imgpath}')
                    # print(f'coco_img_obj={coco_img_obj}')
                    # print('            ')
                    continue


                TEXT_PROMPT = ','.join(det_subset.values())

                image_source, image = load_image(imgpath)

                pred_bboxes, pred_bboxes_unmasked, logits, phrases, sub_tokens, token_logits, token_logits_raw, class_logits, class_logits_raw, mask, tokenidx2class, class2tokenidx = predict(
                    model=model, 
                    bboxes = gt_bboxes,
                    image=image, 
                    caption=TEXT_PROMPT, 
                    box_threshold=BOX_TRESHOLD, 
                    text_threshold=TEXT_TRESHOLD
                )

                assert len(phrases) == len(pred_bboxes), f'{imgpath}'

                ###########################################
                #    For all bboxes in a single frame     #
                ###########################################
                labels_pred = class_logits.max(dim=1).indices.detach().cpu().tolist()
                labels_pred_des = [id2class[i] for i in labels_pred]
                # print(f'cnt={cnt}')
                # print(f'labels_pred={labels_pred}')
                # print(f'labels_pred_des={labels_pred_des}')

                probs = logits_to_probs(class_logits_raw)
                max_probs = probs.max(axis=1).tolist()
                probs = probs.tolist()
                
                probs_sig = logits_to_probs(class_logits)
                max_probs_sig = probs_sig.max(axis=1).tolist()
                probs_sig = probs_sig.tolist()

                class_logits = class_logits.detach().cpu().tolist()
                class_logits_raw = class_logits_raw.detach().cpu().tolist()
                output_pred_bboxes = pred_bboxes.detach().cpu().tolist()

                ###################################################
                #    token_logits_raw.sigmoid() == token_logits   #
                #           if classname is not splitted,         #
                #         then token logit == class_logits        #
                ###################################################
                # print(f'\n----- tokenidx2class -----')
                # pp.pprint(tokenidx2class)
                # print('')
                # print(f'\n----- class2tokenidx -----')
                # pp.pprint(class2tokenidx)
                # print('')


                # print(f'imgpath={imgpath}')
                # print(f'len(gt_bboxes)={len(gt_bboxes)}')
                # print(f'len(truelabels)={len(truelabels)}')
                # print(f'len(truelabels_des)={len(truelabels_des)}')
                # print(f'len(labels_gt)={len(labels_gt)}')
                # print(f'len(labels_gt_des)={len(labels_gt_des)}')
                # print(f'len(labels_pred)={len(labels_pred)}')
                # print(f'len(labels_pred_des)={len(labels_pred_des)}')
                # print(f'len(class_logits)={len(class_logits)}')
                # print(f'len(class_logits_raw)={len(class_logits_raw)}')
                # print(f'len(probs)={len(probs)}')
                # print(f'len(max_probs)={len(max_probs)}')
                # print(f'len(probs_sig)={len(probs_sig)}')
                # print(f'len(max_probs_sig)={len(max_probs_sig)}')

                # Random assertion
                assert len(probs) == len(gt_bboxes)
                assert len(class_logits) == len(gt_bboxes)

                for bbox_gt, bbox_pred, label_gt, label_gt_des, label_pred, label_pred_des, class_logit, class_logit_raw, prob, max_prob, prob_sig, max_prob_sig in zip(gt_bboxes, output_pred_bboxes, labels_gt, labels_gt_des, labels_pred, labels_pred_des, class_logits, class_logits_raw, probs, max_probs, probs_sig, max_probs_sig):
                    
                    bbox_gt_str = ';'.join([str(i) for i in bbox_gt])
                    bbox_pred_str = ';'.join([str(i) for i in bbox_pred])
                    class_logit_str = ';'.join([str(i) for i in class_logit])
                    class_logit_raw_str = ';'.join([str(i) for i in class_logit_raw])
                    prob_str = ';'.join([str(i) for i in prob])
                    prob_sig_str = ';'.join([str(i) for i in prob_sig])

                    row = [
                            cnt,
                            imgpath,
                            bbox_gt_str,
                            bbox_pred_str,
                            label_gt, 
                            label_gt_des, 
                            label_pred, 
                            label_pred_des,
                            class_logit_str,
                            class_logit_raw_str,
                            prob_str,
                            max_prob,
                            prob_sig_str,
                            max_prob_sig
                    ]

                    writer.writerow(row)
                    # print(f'Write {cnt}-th line')
                    cnt += 1

                del pred_bboxes, pred_bboxes_unmasked, logits, phrases, sub_tokens, token_logits, token_logits_raw, class_logits, class_logits_raw, mask, tokenidx2class, class2tokenidx
                del gt_bboxes, probs, max_probs, probs_sig, max_probs_sig

                gc.collect()

            except Exception as e:
                print(f'Error: {e}')
                print(f'Error: {traceback.print_stack()}')
                print(f'cnt={cnt}, imgpath={imgpath}')
                print(f'coco_img_obj={coco_img_obj}')
                print('            ')
                

if __name__ == '__main__':
    import sys
    main(sys.argv[1])