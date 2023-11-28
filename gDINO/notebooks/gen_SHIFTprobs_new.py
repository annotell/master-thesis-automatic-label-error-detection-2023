import os
import gc
import csv
import json
import pickle
import pprint
import itertools
import traceback
import numpy as np
import matplotlib.pyplot as plt
import supervision as sv
from tqdm import tqdm
import torch.nn.functional as F
from tqdm import tqdm

from id_to_class import shift_det as id2class
from groundingdino.util.inference import load_model, load_image, load_copped_image, predict, annotate


class2id = {v: k for k, v in id2class.items()}

pp = pprint.PrettyPrinter(indent=4)


def xyxy2ratio(img_w, img_h, bbox):
    # bbox_left, bbox_top, bbox_width, bbox_height = bbox

    # cx = (bbox_left + bbox_width/2.0)
    # cy = (bbox_top + bbox_height/2.0)

    # cx = 0 if cx < 0 else cx
    # cx = img_w if cx > 0 else cx
    # cy = 0 if cy < 0 else cy
    # cy = img_h if cy > 0 else cy

    # x_center_ratio = cx/img_w
    # y_center_ratio = cx/img_h
    # w_ratio = bbox_width/img_w
    # h_ratio = bbox_height/img_h

    xmin, ymin, xmax, ymax = bbox
    w = xmax-xmin
    h = ymax-ymin
    x_center = (xmin+xmax)/2.0
    y_center = (ymin+ymax)/2.0
    x_center_ratio = x_center/img_w * 1.0
    y_center_ratio = y_center/img_h * 1.0
    w_ratio = w/img_w * 1.0
    h_ratio = h/img_h * 1.0


    return [x_center_ratio, y_center_ratio, w_ratio, h_ratio]


# def gen_id_mapper(all_map, target_map):
#     new_mapper = {}

#     for idx, (ori_classid, classname) in enumerate(all_map.items()):
#         assert idx == ori_classid, f'idx={idx} is not equal to ori_classid={ori_classid}'
#         if classname in target_map.values():
#             targte_id = [k for k, v in target_map.items() if v == classname][0]
#             new_mapper.update({ori_classid: targte_id})
#         else:
#             new_mapper.update({ori_classid:  idx})

#     return new_mapper

def gen_id_mapper(shift_all, shift_target):
    id_mapper = {}
    counter = 0
    shift_target_reverse = {v: k for k, v in shift_target.items()}
    
    # First pass: Map existing items to new values based on shift_target
    for old_id, name in shift_all.items():
        if name in shift_target_reverse:
            id_mapper[old_id] = shift_target_reverse[name]
    
    # Second pass: For remaining items, assign new values
    for old_id, name in shift_all.items():
        if name not in shift_target_reverse:
            while counter in shift_target_reverse.values():
                counter += 1
            id_mapper[old_id] = counter
            counter += 1
            
    return id_mapper


shift_all = {
    0: "pedestrian",
    1: "car",
    2: "truck",
    3: "bus",
    4: "motorcycle",
    5: "bicycle"
}

shift_all_map = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5
}


shift_vehicle = {
    0: "car",
    1: "truck",
    2: "bus",
    # 3: "pedestrian",
    # 4: "motorcycle",
    # 5: "bicycle"
}

shift_vehicle_map2 = {
    1: 0, # original: car(1), new: car(0)
    2: 1, # original: truck(2), new: truck(1)
    3: 2, # original: bus(3), new: bus(2)
    0: 3, # original: pedestrian(0), new: pedestrian(3)
    4: 4, # original: motorcycle(4), new: motorcycle(4)
    5: 5  # original: bicycle(5), new: bicycle(5)
}

shift_vehicle_map = gen_id_mapper(shift_all, shift_vehicle)
assert shift_vehicle_map == shift_vehicle_map2

shift_moto = {
    0: "pedestrian",
    1: "motorcycle",
    2: "bicycle"
    # 3: "car",
    # 4: "truck",
    # 5: "bus",
}

shift_moto_map2 = {
    0: 0, # original: pedestrian(0), new: pedestrian(0)
    4: 1, # original: motorcycle(4), new: motorcycle(1)
    5: 2, # original: bicycle(5), new: bicycle(2)
    1: 3, # original: car(1), new: car(3)
    2: 4, # original: truck(2), new: truck(4)
    3: 5  # original: bus(3), new: bus(5)
}

shift_moto_map = gen_id_mapper(shift_all, shift_moto)
assert shift_moto_map == shift_moto_map2

# heterogeneous classes
shift_heter = {
    # 3: "pedestrian",
    0: "motorcycle",
    # 4: "bicycle"
    1: "car",
    # 5: "truck",
    2: "bus",
}

shift_heter_map = gen_id_mapper(shift_all, shift_heter)

# heterogeneous classes and only two classes
shift_carped = {
    0: "car",
    1: "pedestrian",
    # 2: "truck",
    # 3: "bus",
    # 4: "motorcycle",
    # 5: "bicycle"
}

shift_carped_map = gen_id_mapper(shift_all, shift_carped)

# heterogeneous classes and only two classes
shift_carmo = {
    0: "car",
    1: "motorcycle",
    # 2: "truck",
    # 3: "bus",
    # 4: "pedestrian",
    # 5: "bicycle"
}

shift_carmo_map = gen_id_mapper(shift_all, shift_carmo)

# subset_name = 'all'
# det_subset = shift_all
# det_map = shift_all_map

# subset_name = 'vehicle'
# det_subset = shift_vehicle
# det_map = shift_vehicle_map

# subset_name = 'moto'
# det_subset = shift_moto
# det_map = shift_moto_map

# subset_name = 'mobike'
# det_subset = shift_moto
# det_map = shift_moto_map

# subset_name = 'heter'
# det_subset = shift_heter
# det_map = shift_heter_map

# subset_name = 'carped'
# det_subset = shift_carped
# det_map = shift_carped_map

subset_name = 'carmo'
det_subset = shift_carmo
det_map = shift_carmo_map


def cls_to_probs(cls_scores, gt_bboxes):

    # print(f'[cls_to_probs] cls_score.shape={cls_score.shape}')

    probs = F.softmax(cls_scores, dim=-1).cpu().numpy()
    # print('cls_score.shape=', cls_score.shape)
    # print(f'scores.shape={probs.shape}')
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
    #probs = F.softmax(logits, dim=-1)

    # sanity check
    #close_to_1 = probs.sum(axis=1)
    #print(f'close_to_1={close_to_1}')

    return probs


def main(m_name):

    print(f'Running {m_name}...')

    username = os.getlogin()
    home = f'/home/{username}/Documents/github_others/gDINO'
    pp = pprint.PrettyPrinter(indent=4)
    print(f'HOME={home}')

    weight_name_acronym ='cog'
    weight_name = 'groundingdino_swinb_cogcoor.pth'
    conf_path = f'{home}/groundingdino/config/GroundingDINO_SwinB.cfg.py'

    # weight_name_acronym ='ogc'
    # weight_name = "groundingdino_swint_ogc.pth"
    # conf_path = f'{home}/groundingdino/config/GroundingDINO_SwinT_OGC.py'

    weight_path = os.path.join(home, "notebooks/weights", weight_name)


    print(weight_path, "; exist:", os.path.isfile(weight_path))
    print(conf_path, "; exist:", os.path.isfile(conf_path))

    
    model = load_model(conf_path, weight_path)

    ############################################
    #                 KOGNIC                   #
    ############################################
    # json_name = '3302_annotate'

    # # lambda
    # # image_dir = '/mnt/bfd/yuc/kognic_test/3302_all'
    # # json_path = f'/mnt/bfd/yuc/kognic_test/3302_all_json/{json_name}.json'

    # # yc
    # image_dir = f'{data_dirpath}/kognic/3302_all'
    # json_path = ff'{data_dirpath}/kognic/3302_json/{json_name}.json'
   
    BOX_TRESHOLD = 0.2
    TEXT_TRESHOLD = 0.15

    ############################################
    #                 SHIFT                  #
    ############################################
    json_dirname = 'val'
    # m_name = 'ass1'
    json_name = f'det_2d_stereo_{m_name}'
    image_dir = f'{data_dirpath}/SHIFT/val/RGB_stereo'
    json_path = ff'{data_dirpath}/SHIFT/{json_dirname}/{json_name}.json'

    if weight_name_acronym == 'cog':
        target_csv_dirpath = f'{target_csv_dirpath}/yu/data_pred/SHIFT_gdino_cog'
    elif weight_name_acronym == 'ogc':
        target_csv_dirpath = f'{target_csv_dirpath}/yu/data_pred/SHIFT_gdino_ogc'

    target_csv_path = f'{target_csv_dirpath}/{json_dirname}_{m_name}_box{BOX_TRESHOLD:.1f}_{subset_name}.csv'

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


    print(f'\n========== Process {len(data_dict["frames"])} images ==========')
    print(f'\n========== Process {list(data_dict.keys())} images ==========')

    print(f'Creating csv file: {target_csv_path}')

    if os.path.isfile(target_csv_path):
        print(f'File existed! {target_csv_path}')
        print(f'You will overwrite the file!')

        # Stop all process
        sys.exit()
        return
    

    with open(f'{target_csv_path}', 'w', newline='') as f:
        cnt = 0
        writer = csv.writer(f)
        header_line = ',imgpath,bbox_gt,true_label,true_label_des,label_gt,label_gt_des,label_pred,label_pred_des,class_logit,class_logit_raw,prob,max_prob,prob_sig,max_prob_sig\n'
        f.write(header_line)

       
        for idx, frame in enumerate(tqdm(data_dict['frames'])):
            try:
                imgname = frame['name']
                imgpath = f"{image_dir}/{frame['videoName']}/{frame['name']}"

                assert os.path.isfile(imgpath), f'Not existed! {imgpath}'

                img_shape = (data_dict['config']['imageSize']['width'], data_dict['config']['imageSize']['height'])

                gt_bboxes = []
                truelabels = []
                truelabels_des = []
                labels_gt = []
                labels_gt_des = []

                if len(frame['labels']) == 0:
                    print(f'*** Skip empty frame! ***')
                    print(f'cnt={cnt}, imgpath={imgpath}')
                    print(f'frame={frame}')
                    print(f'labels={frame["labels"]}')
                    print('            ')
                    continue

                for label in frame['labels']:
                    # No numpy or Tensor objects in this loop

                    truelabel_des = label['ori_cat']
                    truelabel = class2id[truelabel_des]

                    label_gt_des = label['category']
                    label_gt = class2id[label_gt_des]

                    if label_gt_des not in list(det_subset.values()):
                        continue
                    
                    new_classid = det_map[label_gt]
                    new_truelabel = det_map[truelabel]

                    bbox = (label['box2d']['x1'], label['box2d']['y1'], label['box2d']['x2'], label['box2d']['y2'])
                    gt_bbox = xyxy2ratio(img_shape[0], img_shape[1], bbox)
                    
                    if len(gt_bbox) ==0:
                        continue

                    gt_bboxes.append(gt_bbox)
                    truelabels.append(truelabel)
                    truelabels_des.append(truelabel_des)
                    labels_gt.append(new_classid)
                    labels_gt_des.append(label_gt_des)

                    # For DEBUG
                    # cnt += 1

                # DEBUG
                # if idx >= 10023:
                #     print(f'*** target frame idx={idx}***')
                #     print(frame['labels'])

                # else:
                #     continue
                    
                TEXT_PROMPT = ','.join(det_subset.values())

                if subset_name == 'mobike':
                    TEXT_PROMPT = TEXT_PROMPT.replace('bicycle', 'bike')

                image_source, image = load_image(imgpath)

                if len(gt_bboxes) == 0:
                    print(f'*** Skip empty frame! ***')
                    continue

                # if cnt ==6:
                #     print(f'cnt={cnt}')

                pred_bboxes, pred_bboxes_unmasked, logits, phrases, sub_tokens, token_logits, token_logits_raw, class_logits, class_logits_raw, mask, tokenidx2class, class2tokenidx = predict(
                    model=model, 
                    bboxes = gt_bboxes,
                    image=image, 
                    caption=TEXT_PROMPT, 
                    box_threshold=BOX_TRESHOLD, 
                    text_threshold=TEXT_TRESHOLD
                )

                assert len(phrases) == len(pred_bboxes), f'{image_path}'
                
                ###########################################
                #    For all bboxes in a single frame     #
                ###########################################
                # print(f'cnt={cnt}')
                labels_pred = class_logits.max(dim=1).indices.detach().cpu().tolist()
                labels_pred_des = [id2class[i] for i in labels_pred]
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

                # debug
                # continue

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
                # # print(f'len(output_pred_bboxes)={len(output_pred_bboxes)}')
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

                # print(f'Inference frame idx={idx}, len of labels_gt_des={len(labels_gt_des)} {labels_gt_des}')

                max_idx_label = None
                for label_idx, (bbox_gt, true_label, true_label_des, label_gt, label_gt_des, label_pred, label_pred_des, class_logit, class_logit_raw, prob, max_prob, prob_sig, max_prob_sig) in enumerate(zip(gt_bboxes, truelabels, truelabels_des, labels_gt, labels_gt_des, labels_pred, labels_pred_des, class_logits, class_logits_raw, probs, max_probs, probs_sig, max_probs_sig)):

                    
                    bbox_gt_str = ';'.join([str(i) for i in bbox_gt])
                    # bbox_pred_str = ';'.join([str(i) for i in bbox_pred])
                    class_logit_str = ';'.join([str(i) for i in class_logit])
                    class_logit_raw_str = ';'.join([str(i) for i in class_logit_raw])
                    prob_str = ';'.join([str(i) for i in prob])
                    prob_sig_str = ';'.join([str(i) for i in prob_sig])

                    row = [
                            cnt,
                            imgpath,
                            bbox_gt_str,
                            true_label,
                            true_label_des,
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
                    cnt += 1
                    max_idx_label = label_idx
                    print(f'idx={idx}, cnt={cnt}, imgpath={imgpath}, len of label_gt_des={label_gt_des}')

                assert label_idx+1 == len(truelabels), f'max label_idx={label_idx}+1 is not equal to len(truelabels)={len(truelabels)}'

                del pred_bboxes, pred_bboxes_unmasked, logits, phrases, sub_tokens, token_logits, token_logits_raw, class_logits, class_logits_raw, mask, tokenidx2class, class2tokenidx
                del gt_bboxes, probs, max_probs, probs_sig, max_probs_sig

                gc.collect()

            except Exception as e:
                print(f'Error: {e}')
                print(f'Error: {traceback.print_stack()}')
                print(f'cnt={cnt}, imgpath={imgpath}')
                print(f'frame={frame}')
                print(f'labels={frame["labels"]}')
                print('            ')
                

if __name__ == '__main__':
    import sys
    main(sys.argv[1])