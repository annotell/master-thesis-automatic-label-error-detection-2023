import os
import json
from collections import Counter 

import numpy as np
import pandas as pd
from PIL import Image
import cv2
from skimage.draw import polygon_perimeter
import matplotlib as mpl
# matplotlib.use('TkAgg')
import torch.nn.functional as F
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from tqdm import tqdm 

from id_to_class import shift_det as id2class
from id_to_class import shift_det_name2RGB as name2RGB
from id_to_class import shift_det_name2red as name2red
from id_to_class import shift_det_name2white as name2white

from tmp_imgpaths import tmp_imgpaths, draw_class, draw_method

def draw_bboxes(fname, img, gt_bboxes, labels_des, name2RGB, isshow, isfont):

    for idx, (gt_bbox, label_des,) in enumerate(zip(gt_bboxes,  labels_des)):

        xmin, ymin, xmax, ymax = [int(i) for i in gt_bbox]

        r = [xmin, xmax, xmax, xmin, xmin]
        c = [ymax, ymax, ymin, ymin, ymax]
        rr, cc = polygon_perimeter(r, c, img.shape)
        # rr, cc = polygon_perimeter(start, end=end, shape=img.shape)
        rgb = name2RGB[label_des]
        img[rr, cc ,0] = rgb[0]
        img[rr, cc ,1] = rgb[1]
        img[rr, cc ,2] = rgb[2]

        if isfont is True:
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_org = (ymin, xmin-2)

            if label_des == 'traffic sign':
                label_des = 'sign'
            elif label_des == 'traffic light':
                label_des = 'light'

            labeltext = f'{label_des}'
            cv2.putText(img, labeltext, text_org, font, 0.5, rgb, 1)

    # print(f'img.shape={img.shape}')
    # img = np.moveaxis(img, 0, -1)
    
    # dpi = 150
    # fig = plt.figure(dpi=dpi, figsize=(12,14))
    # plt.imshow(img)
    if isshow is True:
        dpi = 300
        fig = plt.figure(dpi=dpi, figsize=(7,9))
        plt.imshow(img)
        plt.axis('off')
    
    # plt.savefig(f'{eval_dirpath}/fig/SHIFT/perclass_car/{fname}', bbox_inches='tight')
    
    return img



def stat_coco():

    from id_to_class import coco_shiftall_det as id2class

    class2id = {v: k for k, v in id2class.items()}

    noise_rate = ''
    noise_pattern = 'clean'
    source_anno_path = f'{data_dirpath}/coco2017/annotations/shiftall_val2017_{noise_pattern}{noise_rate}.json'

    with open(source_anno_path, 'r') as f:
        data = json.load(f)

    label_gt_des_list = []
    true_label_des_list = []

    for idx, item in tqdm(enumerate(data['annotations'])):

        label_gt = item['category_id']
        label_gt_des = id2class[label_gt]
        true_label = item['ori_cat']
        true_label_des = id2class[true_label]


        label_gt_des_list.append(label_gt_des)
        true_label_des_list.append(true_label_des)

    cnt_true_label = Counter(true_label_des_list)
    cnt_label_gt = Counter(label_gt_des_list)

    print(f'Noise Pattern: {noise_pattern}{noise_rate}')
    print('cnt_true_label:')
    print(cnt_true_label)
    print(f'cnt_label_gt:')
    print(cnt_label_gt)




def add_noise_coco():

    from id_to_class import coco_shiftall_det as id2class

    peak_mapper = {
        "person": "bicycle",
        "car": "bus",
        "truck": "car",
        "bus": "truck",
        "motorcycle": "bicycle",
        "bicycle": "motorcycle",
    }

    class2id = {v: k for k, v in id2class.items()}

    random_rounds = range(1, 6)

    for round_i in random_rounds:
        noise_rate = 1
        noise_pattern = 'ass'
        source_anno_path = f'{data_dirpath}/coco2017/annotations/shiftall_val2017.json'
        target_anno_dirpath = f'{data_dirpath}/coco2017/annotations/shiftall_{noise_pattern}{noise_rate}'
        target_anno_path = f'{target_anno_dirpath}/shiftall_{noise_pattern}{noise_rate}_round{round_i}.json'

        if not os.path.exists(target_anno_dirpath):
            os.makedirs(target_anno_dirpath)

        with open(source_anno_path) as f:
            data = json.load(f)

        classes = set(list(id2class.values()))

        ori_cla_list = []
        flip_cla_list = []

        for idx, item in tqdm(enumerate(data['annotations'])):

            ori_catid = item['category_id']
            ori_cat = id2class[ori_catid]
            ori_cla_list.append(ori_cat)

            random_num = np.random.uniform(low=0.0, high=1.0, size=None)

            if noise_pattern == 'clean':
                pass

            elif random_num < 0.01 * noise_rate:

                if noise_pattern == 'uni':
                    # Uniform noise
                    other_classes = list(classes - set([ori_cat]))
                    flip_to = np.random.choice(other_classes)

                elif noise_pattern == 'ass':
                    # Asymmetric noise
                    flip_to = peak_mapper[ori_cat]


                flip_to_id = class2id[flip_to]
                flip_cla_list.append(flip_to)
                data['annotations'][idx]['category_id'] = flip_to_id

            data['annotations'][idx]['ori_cat'] = ori_catid


        with open(target_anno_path, "w") as outfile:
            json.dump(data, outfile)

        cnt = Counter(ori_cla_list)
        cnt_flip = Counter(flip_cla_list)

        print(f'Noise Pattern: {noise_pattern}{noise_rate}')
        print(cnt)
        print(cnt_flip)


def main():

    random_rounds = range(1, 6)

    for round_i in random_rounds:
        noise_rate = 5
        data_type = 'val'
        noise_type = 'uni'
        dirpath = f'{data_dirpath}/SHIFT/{data_type}/det_2d_stereo_clean.json'

        target_dirpath = f'{data_dirpath}/SHIFT/{data_type}_stereo_rounds/{noise_type}{noise_rate}'
        target_filepath = f'{target_dirpath}/{noise_type}{noise_rate}_round{round_i}.json'
        
        if not os.path.exists(target_dirpath):
            os.makedirs(target_dirpath)


        # ass
        peak_mapper = {
            "pedestrian": "bicycle",
            "car": "bus",
            "truck": "car",
            "bus": "truck",
            "motorcycle": "bicycle",
            "bicycle": "motorcycle",
        }

        with open(dirpath) as f:
            data = json.load(f)

        classes = set(["pedestrian", "car", "truck", "bus", "motorcycle", "bicycle"])

        
        print(len(data['frames']))

        ori_cla_list = []
        flip_cla_list = []
        for frame_idx, frame in enumerate(data['frames']):
            for label_idx, label in enumerate(frame['labels']):

                ori_cla_list.append(label['category'])
                random_num = np.random.uniform(low=0.0, high=1.0, size=None)

                data['frames'][frame_idx]['labels'][label_idx]['ori_cat'] = label['category']

                if random_num < 0.01 * noise_rate:

                    if noise_type == 'uni':
                        # Uniform noise
                        other_classes = list(classes - set([label['category']]))
                        flip_to = np.random.choice(other_classes)

                    elif noise_type == 'gau':
                        # Uniform noise
                        other_classes = list(classes - set([label['category']]))
                        flip_to = np.random.choice(other_classes)
                        

                    elif noise_type == 'ass':
                        # Assymetric noise
                        flip_to = peak_mapper[label['category']]

                    data['frames'][frame_idx]['labels'][label_idx]['category'] = flip_to

                flip_cla_list.append(data['frames'][frame_idx]['labels'][label_idx]['category'])


        with open(target_filepath, "w") as outfile:
            json.dump(data, outfile)

        cnt = Counter(ori_cla_list)
        cnt_flip = Counter(flip_cla_list)
        print(cnt)
        print(cnt_flip)


def draw_bbox_seq():
    data_type = 'val'
    # source_dirpath = f'{data_dirpath}/SHIFT/{data_type}/det_2d.json'
    # source_dirpath = f'{data_dirpath}/SHIFT/{data_type}/det_2d_stereo_clean.json'
    source_dirpath = f'{data_dirpath}/SHIFT/{data_type}/det_2d_stereo_ass1.json'

    with open(source_dirpath) as f:
        print(f'Reading.. {source_dirpath}')
        data = json.load(f)

    # classes = ["pedestrian", "car", "truck", "bus", "motorcycle", "bicycle"]
    img_dirpath = f'{data_dirpath}/SHIFT/val/RGB_stereo'

    for frame_idx, frame in enumerate(data['frames']):
        imgpath = f'{img_dirpath}/{frame["videoName"]}/{frame["name"]}'

        targets = ['d96a-80f9']


        if frame["videoName"] not in  targets:
            continue

        labels_gt_des = []
        true_labels_des = []
        gt_bboxes = []


        for label_idx, label in enumerate(frame['labels']):
            label_gt_des= label['category']
            true_label_des = label['ori_cat']


            xmin, ymin, xmax, ymax = label['box2d']['x1'], label['box2d']['y1'], label['box2d']['x2'], label['box2d']['y2']

            labels_gt_des.append(label_gt_des)
            true_labels_des.append(true_label_des)
            gt_bboxes.append([ymin, xmin, ymax, xmax])

            # print( true_label_des, label_gt_des)
            print('')
        

        classname = draw_class  # annotated label
        method = draw_method

        # name2white[classname] = (255, 0, 0)


        isshow = True
        isfont = True
        img = np.array(Image.open(imgpath))
        img_shape = img.shape
        fname_gt = f'gt_{frame["videoName"]}_{frame["name"]}'
        img_with_bbox_gt = draw_bboxes(fname_gt, img, gt_bboxes, labels_gt_des, name2white, isshow, isfont)

        img = np.array(Image.open(imgpath))
        img_shape = img.shape
        fname_true = f'true_{frame["videoName"]}_{frame["name"]}'
        img_with_bbox_true = draw_bboxes(fname_true, img, gt_bboxes, true_labels_des, name2white, isshow, isfont)

        img = np.array(Image.open(imgpath))
        img_shape = img.shape
        fname_plain = f'plain_{frame["videoName"]}_{frame["name"]}'
        img_with_bbox_plain = draw_bboxes(fname_true, img, gt_bboxes, true_labels_des, name2white, isshow, isfont=False)

        print('')

        # fig_dirpath = f'{eval_dirpath}/fig/SHIFT_rcnn/gt/{method}_perclass_{classname}'

        fig_dirpath = f'{data_dirpath}/SHIFT/val/RGB_stereo/{frame["videoName"]}_vis'

        if not os.path.exists(fig_dirpath):
            os.makedirs(fig_dirpath)

        plt.imshow(img_with_bbox_gt, cmap="gray")
        plt.axis('off')
        plt.savefig(f"{fig_dirpath}/{fname_gt}", bbox_inches='tight', dpi=500)
        print(f'labels_gt_des={labels_gt_des}')
        print(f'Saved to {fig_dirpath}/{fname_gt}')

        plt.imshow(img_with_bbox_true, cmap="gray")
        plt.axis('off')
        plt.savefig(f"{fig_dirpath}/{fname_true}", bbox_inches='tight', dpi=500)
        print(f'true_labels_des={true_labels_des}')
        print(f'Saved to {fig_dirpath}/{fname_true}')

        plt.imshow(img_with_bbox_plain, cmap="gray")
        plt.axis('off')
        plt.savefig(f"{fig_dirpath}/{fname_plain}", bbox_inches='tight', dpi=500)
        print(f'true_labels_des={true_labels_des}')
        print(f'Saved to {fig_dirpath}/{fname_plain}')


def draw_bbox():
    data_type = 'val'
    # source_dirpath = f'{data_dirpath}/SHIFT/{data_type}/det_2d.json'
    # source_dirpath = f'{data_dirpath}/SHIFT/{data_type}/det_2d_stereo_clean.json'
    source_dirpath = f'{data_dirpath}/SHIFT/{data_type}/det_2d_stereo_ass1.json'

    with open(source_dirpath) as f:
        print(f'Reading.. {source_dirpath}')
        data = json.load(f)

    # classes = ["pedestrian", "car", "truck", "bus", "motorcycle", "bicycle"]
    img_dirpath = f'{data_dirpath}/SHIFT/val/RGB_stereo'

    for frame_idx, frame in enumerate(data['frames']):
        imgpath = f'{img_dirpath}/{frame["videoName"]}/{frame["name"]}'

        targets = tmp_imgpaths

        if imgpath not in  targets:
            continue

        

        labels_gt_des = []
        true_labels_des = []
        gt_bboxes = []


        for label_idx, label in enumerate(frame['labels']):
            label_gt_des= label['category']
            true_label_des = label['ori_cat']


            xmin, ymin, xmax, ymax = label['box2d']['x1'], label['box2d']['y1'], label['box2d']['x2'], label['box2d']['y2']

            labels_gt_des.append(label_gt_des)
            true_labels_des.append(true_label_des)
            gt_bboxes.append([ymin, xmin, ymax, xmax])

            # print( true_label_des, label_gt_des)
            print('')
        

        classname = draw_class  # annotated label
        method = draw_method

        name2white[classname] = (255, 0, 0)


        isshow = True
        isfont = True
        img = np.array(Image.open(imgpath))
        img_shape = img.shape
        fname_gt = f'gt_{frame["videoName"]}_{frame["name"]}'
        img_with_bbox_gt = draw_bboxes(fname_gt, img, gt_bboxes, labels_gt_des, name2white, isshow, isfont)

        img = np.array(Image.open(imgpath))
        img_shape = img.shape
        fname_true = f'true_{frame["videoName"]}_{frame["name"]}'
        img_with_bbox_true = draw_bboxes(fname_true, img, gt_bboxes, true_labels_des, name2white, isshow, isfont)

        img = np.array(Image.open(imgpath))
        img_shape = img.shape
        fname_plain = f'plain_{frame["videoName"]}_{frame["name"]}'
        img_with_bbox_plain = draw_bboxes(fname_true, img, gt_bboxes, true_labels_des, name2white, isshow, isfont=False)

        print('')

        fig_dirpath = f'{eval_dirpath}/fig/SHIFT_rcnn/gt/{method}_perclass_{classname}'

        if not os.path.exists(fig_dirpath):
            os.makedirs(fig_dirpath)

        plt.imshow(img_with_bbox_gt, cmap="gray")
        plt.axis('off')
        plt.savefig(f"{fig_dirpath}/{fname_gt}", bbox_inches='tight', dpi=300)
        print(f'labels_gt_des={labels_gt_des}')
        print(f'Saved to {fig_dirpath}/{fname_gt}')

        plt.imshow(img_with_bbox_true, cmap="gray")
        plt.axis('off')
        plt.savefig(f"{fig_dirpath}/{fname_true}", bbox_inches='tight', dpi=300)
        print(f'true_labels_des={true_labels_des}')
        print(f'Saved to {fig_dirpath}/{fname_true}')

        plt.imshow(img_with_bbox_plain, cmap="gray")
        plt.axis('off')
        plt.savefig(f"{fig_dirpath}/{fname_plain}", bbox_inches='tight', dpi=300)
        print(f'true_labels_des={true_labels_des}')
        print(f'Saved to {fig_dirpath}/{fname_plain}')



def seperate_class():

    noise_rate = 1
    data_type = 'val'
    source_dirpath = f'{data_dirpath}/coco2017/annotations/instances_{data_type}2017.json'

    with open(source_dirpath) as f:
        data = json.load(f)

    # SHIFT
    # classes = ["pedestrian", "car", "truck", "bus", "motorcycle", "bicycle"]
    
    # COCO
    coco_shiftall = {
        0: "car",
        1: "truck",
        2: "bus",
        3: "person",
        4: "motorcycle",
        5: "bicycle"
    }
    classes = [i for i in coco_shiftall.values()]

    gen_stat(data, classes, is_latex=True)

    # classes = set(classes)

    # class1 = set(["pedestrian", "car", "truck"])
    # class2 = set(["bus", "motorcycle", "bicycle"])

    
    # print(len(data['frames']))

    # ori_cla_list = []
    # flip_cla_list = []

    # data_class1 = select_classes(data, class1)
    # data_class2 = select_classes(data, class2)

    # gen_stat(data_class1)
    # gen_stat(data_class2)


    # data_class1 = add_uniform_noise(data_class1, class1, noise_rate)
    # data_class2 = add_uniform_noise(data_class2, class2, noise_rate)

    # print(f'After Adding Uniform Noise')
    # gen_stat(data_class1)
    # gen_stat(data_class2)

    # with open(target_jsonpath_class1, "w") as outfile:
    #     json.dump(data_class1, outfile)

    # with open(target_jsonpath_class2, "w") as outfile:
    #     json.dump(data_class1, outfile)



def add_uniform_noise(data, default_classes, noise_rate):

    # original_classes= []
    # flip_classes = []

    for frame_idx, frame in enumerate(data['frames']):
        for label_idx, label in enumerate(frame['labels']):

            # original_classes.append(label['category'])
            random_num = np.random.uniform(low=0.0, high=1.0, size=None)

            data['frames'][frame_idx]['labels'][label_idx]['ori_cat'] = label['category']

            if random_num < 0.01 * noise_rate:

                # Uniform noise
                other_classes = list(default_classes - set([label['category']]))
                flip_to = np.random.choice(other_classes)

                data['frames'][frame_idx]['labels'][label_idx]['category'] = flip_to

            # flip_classes.append(data['frames'][frame_idx]['labels'][label_idx]['category'])
    
    return data



def select_classes(data, selected_class):

    data_class = {
        'frames': [],
        'config': {
            "imageSize": {
                "width": 1280,
                "height": 800
            },
            "categories": []
        }
    }

    for classname in selected_class:
        data_class['config']['categories'].append({"name": classname})

    for frame_idx, frame in enumerate(data['frames']):

        new_frame = frame.copy()
        new_frame['labels'] = []

        for label_idx, label in enumerate(frame['labels']):

            if label['category'] in selected_class:

                new_frame['labels'].append(label)
            
        data_class['frames'].append(new_frame)

    return data_class


def percentage_info(counter):
    total_count = sum(counter.values())
    return {key: f'{((value / total_count) * 100):.2f}' for key, value in counter.items()}


def gen_stat(data, input_classes, is_latex=False):

    ori_classes = []
    classes = []

    for frame_idx, frame in enumerate(data['frames']):
        for label_idx, label in enumerate(frame['labels']):
            classes.append(label['category'])

            if 'ori_cat' in label.keys():
                ori_classes.append(label['ori_cat'])
    
    ori_cnt = Counter(ori_classes)
    cnt = Counter(classes)

    print(f'Original classes:')
    print(ori_cnt)
    print(percentage_info(ori_cnt))
    print(f'classes: ')
    print(cnt)
    print(percentage_info(cnt))
    print(f'=================================')

    if is_latex:
        print(f'latent prior:')
        df1 = pd.DataFrame.from_dict(ori_cnt, columns=['count'], orient='index')
        df1 = df1.reindex(input_classes)
        df1['percent'] = df1['count'] / df1['count'].sum() * 100.0
        df1['percent'] = df1['percent'].apply(lambda x: f'{x:.2f}')
        print(df1)
        latex_tbl = df1.style.to_latex()
        print(latex_tbl)
        print(f'=================================')
        print(f'Observed prior: ')
        df2 = pd.DataFrame.from_dict(cnt, columns=['count'], orient='index')
        df2 = df2.reindex(input_classes)
        df2['percent'] = df2['count'] / df2['count'].sum() * 100.0
        df2['percent'] = df2['percent'].apply(lambda x: f'{x:.2f}')
        print(df2)
        latex_tbl = df2.style.to_latex()
        print(latex_tbl)



if __name__ == '__main__':
    # main()
    # seperate_class()
    # draw_bbox()
    draw_bbox_seq()
    # add_noise_coco()
    # stat_coco()