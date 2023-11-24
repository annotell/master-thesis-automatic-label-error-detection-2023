import os
import csv
import json
import pickle
import pprint
import traceback
import torch
import numpy as np
import matplotlib.pyplot as plt
import supervision as sv
from tqdm import tqdm

from id_to_class import bdd100k_det as id2class
# from id_to_class import bdd100k_det_name2RGB as name2RGB
# from id_to_class import bdd100k_det_name2white as name2white

# from id_to_class import shift_det as id2class
# from id_to_class import shift_det_name2RGB as name2RGB
# from id_to_class import shift_det_name2white as name2white

pp = pprint.PrettyPrinter(indent=4)


def xywh2ratio(img_w, img_h, bbox):
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

    x, y, w, h = bbox
    x_center = x + w/2.0
    y_center = y + h/2.0
    x_center_ratio = x_center/img_w * 1.0
    y_center_ratio = y_center/img_h * 1.0
    w_ratio = w/img_w * 1.0
    h_ratio = h/img_h * 1.0


    return [x_center_ratio, y_center_ratio, w_ratio, h_ratio]


nuimage_all = {
    0: 'car', 
    1: 'truck', 
    2: 'trailer', 
    3: 'bus', 
    4: 'construction_vehicle', 
    5: 'bicycle', 
    6: 'motorcycle', 
    7: 'pedestrian', 
    8: 'traffic_cone', 
    9: 'barrier'
}

nuimage_all_map = {
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
bdd100k_det_subset = bdd100k_all
bdd100k_det_map = bdd100k_all_map

# subset_name = 'traffic'
# bdd100k_det_subset = bdd100k_traffic
# bdd100k_det_map = bdd100k_traffic_map

# subset_name = 'vehicle'
# bdd100k_det_subset = bdd100k_vehicle
# bdd100k_det_map = bdd100k_vehicle_map

# subset_name = 'people'
# bdd100k_det_subset = bdd100k_people
# bdd100k_det_map = bdd100k_people_map


def main():

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

    from groundingdino.util.inference import load_model, load_image, load_copped_image, predict, annotate

    saved_objs = []
    model = load_model(conf_path, weight_path)

    ############################################
    #                 KOGNIC                   #
    ############################################
    # json_name = '3302_annotate'

    # # lambda
    # # image_dir = '/mnt/bfd/yuc/kognic_test/3302_all'
    # # json_path = f'/mnt/bfd/yuc/kognic_test/3302_all_json/{json_name}.json'

    # # yc
    # image_dir = '/media/18T/data_thesis/kognic/3302_all'
    # json_path = f'/media/18T/data_thesis/kognic/3302_json/{json_name}.json'
   
    ############################################
    #                 NuImage                  #
    ############################################
    # json_dirname = 'nuimages-v1.0_coco'
    # json_name = 'nuimages_v1.0-val'
    # image_dir = '/media/18T/data_thesis/NuImages/nuimages-v1.0-all'
    # json_path = f'/media/18T/data_thesis/NuImages/{json_dirname}/{json_name}.json'

    ############################################
    #                 BDD100K                  #
    ############################################
    # json_dirname = 'labels_coco2'
    # json_name = 'train_cocofmt'
    # image_dir = f'{data_dirpath}/bdd100k_images_100k/images/100k/train'

    # json_dirname = 'labels_coco2'
    # json_name = 'val_cocofmt'
    # image_dir = f'{data_dirpath}/bdd100k_images_100k/images/100k/val'

    # json_dirname = 'ShuffleSplit10'
    # json_name = 'Split-1_val'
    # image_dir = f'{data_dirpath}/bdd100k_images_100k/images/100k/train'

    json_dirname = 'OfficialVal_ShuffleSplit10'
    json_name = 'Split-0'
    image_dir = f'{data_dirpath}/bdd100k_images_100k/images/100k/val'

    # Final json path
    json_path = f'{data_dirpath}/bdd100k_images_100k/{json_dirname}/{json_name}.json'


    BOX_TRESHOLD = 0.1
    TEXT_TRESHOLD = 0.1
    # pkl_dir = f'/home/{username}/Documents/github_others/gDINO/injects'

    with open(json_path, 'r') as f:
        data_dict = json.load(f)

    # BDD100k
    # pkl_path = f'{pkl_dir}/{json_dirname}_{json_name}_box{BOX_TRESHOLD:.1f}_{subset_name}_properties.pkl'

    # KOGNIC
    # kognic_id2class = {int(k): v for k, v in data_dict['id2class'].items()}
    # pp.pprint(kognic_id2class)

    # kognic_carvan = {
    #     0: "car",
    #     1: "van"
    # }

    # kognic_trambus = {
    #     0: "tram",
    #     1: "bus"
    # }

    # kognic_vansuvtruck = {
    #     0: "van",
    #     1: "suv",
    #     2: "truck"
    # }

    # kognic_carbikemoto = {
    #     0: "car",
    #     1: "bicycle",
    #     2: "motorcycle"
    # }

    # kognic_vehicle = {
    #     0: "car",
    #     1: "truck",
    #     2: "bus",
    #     3: "tram",
    #     # 4: "suv",
    #     # 5: "trailer",
    #     # 6: "stroller",
    #     # 7: "van"
    # }

    # kognic_person = {
    #     0: "bike",
    #     1: "motorcycle",
    #     # 2: "pedestrian"
    # }


    def create_idx_mapping(oriset, subset):
        new_classnames = list(subset.values())
        ori2subset = {}
        for k, v in oriset.items():
            if v in new_classnames:
                new_idx = new_classnames.index(v)
                ori2subset[k] = new_idx
        return ori2subset

    ################# Change these three line #################
    # kognic_map = create_idx_mapping(kognic_id2class, kognic_carbikemoto)
    # subset_name = 'carbikemoto'
    # bdd100k_det_subset = kognic_carbikemoto
    ###########################################################

    # bdd100k_det_map = kognic_map
    # id2class = bdd100k_vehicle


    # pkl_path = f'{pkl_dir}/{weight_name_acronym}_{json_name}_box{BOX_TRESHOLD:.1f}_{subset_name}.pkl'




    print(f'\n========== Process {len(data_dict["images"])} images ==========')


    csv_path = f'{target_csv_dirpath}/yu/data_pred/bdd100k/{json_dirname}_{json_name}.csv'
    print(f'Creating csv file: {csv_path}')

    with open(f'{csv_path}', 'w', newline='') as f:
        cnt = 0
        writer = csv.writer(f)
        header_line = ',imgpath,cls_score,prob,max_prob,bbox_gt,true_label,true_label_des,label_gt,label_gt_des,label_pred,label_pred_des,loss\n'
        f.write(header_line)

        for idx, coco_img_obj in tqdm(enumerate(data_dict['images'])):

            # Testing
            # if idx == 1:
            #     break

            saved_obj = {
                # 'img': None,
                'img_shape': (coco_img_obj['height'], coco_img_obj['width']),
                # 'img_feature': None,
                'img_path': os.path.join(image_dir, coco_img_obj['file_name']),
                'img_name': coco_img_obj['file_name'],
                'gt_bboxes': [],
                'gt_labels': [],
                'gt_labels_des': [],
                'phrases': None,
                'sub_tokens': None,
                'token_logits': None,
                'token_logits_raw': None,
                'class_logits': None,
                'class_logits_raw': None,
                'pred_bboxes': None,
                'pred_bboxes_unmasked': None
                # 'pred_labels': None,
                # 'pred_labels_des': None,
            }



            image_name = coco_img_obj['file_name']
            #print(f'Process {image_name}')
            # if image_name != 'b38f59d4-8dfeca9f.jpg':
            #     continue

            for anno in data_dict['annotations']:

                if coco_img_obj['id'] == anno['image_id']:

                    bbox = xywh2ratio(coco_img_obj['width'], coco_img_obj['height'], anno['bbox'])
                    
                    # bdd100k
                    classid = anno["category_id"]-1

                    # id2class is the original classes for the json file
                    classname = id2class[classid] # use all to map the original coco format
                    

                    if classname not in list(bdd100k_det_subset.values()):
                        continue
                    
                    new_classid = bdd100k_det_map[classid]
                    saved_obj['gt_bboxes'].append(bbox)
                    saved_obj['gt_labels'].append(new_classid)
                    saved_obj['gt_labels_des'].append(classname)

            

            if len(saved_obj['gt_bboxes']) == 0:
                continue

            image_path = saved_obj['img_path']

            TEXT_PROMPT = ','.join(bdd100k_det_subset.values())


            # cropped bbox
            pad = 4
            image_source, image_transform, cropped_bboxes, bbox_coords = load_copped_image(image_path, saved_obj['gt_bboxes'], pad)
            image_transform = np.swapaxes(image_transform, 0, -1)

            c_bboxes = [np.swapaxes(i, 0, 2) for i in cropped_bboxes]
            c_bboxes = [np.swapaxes(i, 1, 2) for i in c_bboxes]

            # print(f'image.shape={image.shape}')
            print(f'image_path={image_path}')
            assert len(cropped_bboxes) == len(bbox_coords), f'cropped_bboxes={len(cropped_bboxes)}, bbox_coords={len(bbox_coords)}'
            assert len(cropped_bboxes) == len(saved_obj['gt_labels'])
            assert len(cropped_bboxes) == len(saved_obj['gt_labels_des'])

            for idx, (gt_bbox, gt_label, gt_label_des, c_bbox) in enumerate(zip(bbox_coords, saved_obj['gt_labels'], saved_obj['gt_labels_des'], c_bboxes)):
                # try:

                c_bbox = torch.Tensor(c_bbox)
                print(f'c_bbox={c_bbox.shape}')
                print(f'gt_bbox={gt_bbox}')
                print(f'gt_label={gt_label}')
                print(f'gt_label_des={gt_label_des}')

                pred_bboxes, pred_bboxes_unmasked, logits, phrases, sub_tokens, token_logits, token_logits_raw, class_logits, class_logits_raw, mask, tokenidx2class, class2tokenidx = predict(
                    model=model, 
                    bboxes = [gt_bbox],
                    image=c_bbox, 
                    caption=TEXT_PROMPT, 
                    box_threshold=BOX_TRESHOLD, 
                    text_threshold=TEXT_TRESHOLD
                )

                print(f'pred_bboxes={pred_bboxes.shape}')
                print(f'pred_bboxes_unmasked={pred_bboxes_unmasked.shape}')
                print(f'logits={logits.shape}')
                print(f'phrases={phrases}')
                print(f'sub_tokens={sub_tokens}')
                print(f'token_logits={token_logits.shape}')
                print(f'token_logits_raw={token_logits_raw.shape}')
                print(f'class_logits={class_logits.shape}')
                print(f'class_logits_raw={class_logits_raw.shape}')
                print(f'mask={len(mask)}')
                print(f'tokenidx2class={tokenidx2class}')
                print(f'class2tokenidx={class2tokenidx}')
                
                print('-------------')
                # except Exception as e:
                #     print(traceback.print_exc())
                #     plt.imshow(cropped_bboxes[idx])
                #     continue
                
                # # print(f'Original gt_bboxes = {len(saved_obj["gt_bboxes"])}')
                # # print(f'Original gt_labels = {len(saved_obj["gt_labels"])}')
                # # print(f'Original pred_bboxes.shape = {pred_bboxes.shape}')
                # assert len(phrases) == len(pred_bboxes), f'{image_path}'

                # if '' in phrases:
                #     print(phrases)
                # saved_obj['phrases'] = phrases
                # saved_obj['token_logits'] = token_logits
                # saved_obj['token_logits_raw'] = token_logits_raw
                # #saved_obj['img_shape'] = [i for i, m in zip(saved_obj['img_shape'], mask) if m is True]
                # #saved_obj['img_path'] = [i for i, m in zip(saved_obj['img_path'], mask) if m is True]
                # #saved_obj['img_name'] = [i for i, m in zip(saved_obj['img_name'], mask) if m is True]
                # #saved_obj['token_logits'] = [i for i, m in zip(saved_obj['token_logits'], mask) if m is True]
                # #saved_obj['phrases'] = [i for i, m in zip(saved_obj['phrases'], mask) if m is True]
                # #saved_obj['gt_bboxes'] = [i for i, m in zip(saved_obj['gt_bboxes'], mask) if m is True]
                # #saved_obj['gt_labels'] = [i for i, m in zip(saved_obj['gt_labels'], mask) if m is True]
                # #saved_obj['gt_labels_des'] = [i for i, m in zip(saved_obj['gt_labels_des'], mask) if m is True]

                # # print(f'Filtered gt_bboxes = {len(saved_obj["gt_bboxes"])}, Filtered gt_labels = {len(saved_obj["gt_labels"])}')
                # # print(f'Return class_logits = {class_logits.shape}')

                # saved_obj['pred_bboxes'] = pred_bboxes
                # saved_obj['pred_bboxes_unmasked'] = pred_bboxes_unmasked
                # #saved_obj['sub_tokens'] = sub_tokens
                # #saved_obj['token_logits'] = token_logits

                # # use class logits or not
                # saved_obj['class_logits'] = class_logits
                # saved_obj['class_logits_raw'] = class_logits_raw
                # assert len(saved_obj['phrases']) == pred_bboxes.shape[0]
                # assert len(saved_obj['gt_bboxes']) == pred_bboxes_unmasked.shape[0]
                # assert len(saved_obj['class_logits']) == pred_bboxes_unmasked.shape[0]
                # assert len(saved_obj['class_logits_raw']) == pred_bboxes_unmasked.shape[0]

                # if len(saved_obj['gt_bboxes']) == 0 or pred_bboxes.shape[0] == 0:
                #     continue

                # print('here')
                # continue
                # # print(f"[Saved] gt_labels[0]={saved_obj['gt_labels'][0]}")
                # # print(f"[Saved] gt_labels_des[0]={saved_obj['gt_labels_des'][0]}")
                # # print(f"[Saved] gt_bboxes[0]={saved_obj['gt_bboxes'][0]}")
                # # print(f"[Saved] class_logits[0]={saved_obj['class_logits'][0]}")
                # # print(f"[Saved] pred_bboxes[0]={saved_obj['pred_bboxes'][0]}")

                # saved_objs.append(saved_obj)

            # print(f'\n----- tokenidx2class -----')
            # pp.pprint(tokenidx2class)
            # print('')
            # print(f'\n----- class2tokenidx -----')
            # pp.pprint(class2tokenidx)
            # print('')

        # with open(f'{pkl_path}', 'wb') as handle:
        #     pickle.dump(saved_objs, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()