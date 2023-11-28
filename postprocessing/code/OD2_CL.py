import os
import io
import gc
import csv
import time
import pickle
import pprint
import resource
from collections import Counter
from tqdm import tqdm
from multiprocessing import Manager, Process, Pool
from functools import partial
from itertools import compress, chain, repeat
from operator import itemgetter
import operator as op

from sklearn.preprocessing import normalize

import numpy as np
import pandas as pd
from PIL import Image
import cv2
from skimage.draw import polygon_perimeter
import matplotlib as mpl
# matplotlib.use('TkAgg')
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.stats import entropy, norm, uniform
from torchvision import datasets, transforms
from scipy.special import softmax

import faiss
from cleanlab.classification import CleanLearning
from cleanlab.filter import find_label_issues, find_predicted_neq_given, find_label_issues_using_argmax_confusion_matrix
from cleanlab.dataset import health_summary
from cleanlab.count import get_confident_thresholds, estimate_joint, compute_confident_joint, estimate_latent, estimate_noise_matrices, calibrate_confident_joint, estimate_py_and_noise_matrices_from_probabilities
from cleanlab.rank import get_label_quality_scores, get_normalized_margin_for_each_label, get_self_confidence_for_each_label, get_confidence_weighted_entropy_for_each_label
from cleanlab.internal.label_quality_utils import (
    _subtract_confident_thresholds,
    get_normalized_entropy,
)

from model import Net, GtsrbFolderWithPaths, SwedishFolderWithPaths

# from id_to_class import bdd100k_det as id2class
# from id_to_class import bdd100k_det_name2RGB as name2RGB
# from id_to_class import bdd100k_det_name2white as name2white

# from id_to_class import shift_det as id2class
# from id_to_class import shift_det_name2RGB as name2RGB
# from id_to_class import shift_det_name2white as name2white

import torch
pp = pprint.PrettyPrinter(indent=4)
device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")

# if 6 in id2class.keys():
#     del id2class[6]



ranked_by_options = ['self_confidence', 'normalized_margin', 'confidence_weighted_entropy']
filter_by_options = ['prune_by_class', 'prune_by_noise_rate', 'both', 'confident_learning', 'predicted_neq_given']
shorten = {
    'normalized_margin': 'NM',
    'self_confidence': 'SC',
    'confidence_weighted_entropy': 'CWE',
    'prune_by_class': 'PBC',
    'prune_by_noise_rate': 'PBNR',
    'both': 'BO',
    'confident_learning': 'CL',
    'predicted_neq_given':  'PNG',
    'Loss_Ranking': 'LRank',
    'normalized_margin': 'NM',
    'self_confidence': 'SC', 
    'relative_confidence': 'RC',
    'entropy': 'EN',
    'relative_entropy': 'REN'
}


def ratio_cxcywh2xyxy(img_w, img_h, bbox):

    cx, cy, w, h = bbox
    # x is row
    # y is column
    # print('[original ratio_cxcywh2xyxy] img_w, img_h', img_w, img_h)
    # print('[original ratio_cxcywh2xyxy] ', cx, cy, w, h)
    
    xmin = img_w*(cx - w/2.0)
    ymin = img_h*(cy - h/2.0)
    xmax = img_w*(cx + w/2.0)
    ymax = img_h*(cy + h/2.0)
    
    xmin = 0 if xmin < 0 else xmin
    ymin = 0 if ymin < 0 else ymin
    xmax = xmax if xmax < img_w else img_w
    ymax = ymax if ymax < img_h else img_h
    
    # print('[after ratio_cxcywh2xyxy] ', xmin, ymin, xmax, ymax)
    return xmin, ymin, xmax, ymax


def plot_issues(df, dataset, method, name2RGB, name2red, fig_filepath, show_img=False):

    dpi = 400

    # big and fit!
    # fig = plt.figure(dpi=dpi, figsize=(20,40))
    # nrows = 40
    # ncols = 20

    # small and fit!
    fig = plt.figure(dpi=dpi, figsize=(20,10))
    nrows = 5
    ncols = 10

    start = 0
    end = start + nrows * ncols

    topn = nrows * ncols
    df = df.sort_values(method, ascending=True, inplace=False)
    df = df[start:end]
    cnt = 0
    
    for bbox_idx, row in df.iterrows():
        # print(f'[def plot_issues] imgpath={row["imgpath"]}')
        # print(f'[def plot_issues] label_gt_des={row["label_gt_des"]}')
        # print(f'[def plot_issues] label_pred_des={row["label_pred_des"]}')
        img = np.array(Image.open(row['imgpath']))
        img_shape = img.shape

        if img.ndim ==2:
            print(f'Convert gray image to RGB')
            # Load the grayscale image 
            img_gray = Image.open(row['imgpath']).convert('L') 
            
            # Create a new RGB image with three identical grayscale channels 
            img = Image.merge('RGB', [img_gray]*3)
            img = np.array(img)


        isshow = False
        isfont = False

        if dataset == 'bdd100k':
            # bdd100k don't not have true_label, use c_incorrect from default_df
            img_with_bbox = draw_bboxes(dataset, img, [row['bbox_gt']], [row['label_gt']], [row['label_gt_des']], [row['label_pred']], [row['label_pred_des']], [row['c_incorrect']], name2RGB, name2red, isshow, isfont)
            # plt.imshow(img_with_bbox, cmap="gray")
            bbox_img = crop_bboxes(img_with_bbox, img_shape, [row['bbox_gt']], [row['label_gt']], pad=15)[0]
            title = f'Anno={row["label_gt_des"]}\nPred={row["label_pred_des"]}'

        elif dataset == 'bdd100k_gdino':
            # bdd100k don't not have true_label, use c_incorrect from default_df
            img_with_bbox = draw_bboxes(dataset, img, [row['bbox_gt']], [row['label_gt']], [row['label_gt_des']], [row['label_pred']], [row['label_pred_des']], [row['c_incorrect']], name2RGB, name2red, isshow, isfont)
            # plt.imshow(img_with_bbox, cmap="gray")
            bbox_img = crop_bboxes(img_with_bbox, img_shape, [row['bbox_gt']], [row['label_gt']], pad=15)[0]
            title = f'Anno={row["label_gt_des"]}\nPred={row["label_pred_des"]}'

        else:
            assert dataset in ['shift', 'coco']
            img_with_bbox = draw_bboxes(dataset, img, [row['bbox_gt']], [row['label_gt']], [row['label_gt_des']], [row['label_pred']], [row['label_pred_des']], [row['true_label_des']], name2RGB, name2red, isshow, isfont)
            # plt.imshow(img_with_bbox, cmap="gray")
            bbox_img = crop_bboxes(img_with_bbox, img_shape, [row['bbox_gt']], [row['label_gt']], pad=15)[0]
            title = f'True={row["true_label_des"]}\nAnno={row["label_gt_des"]}\nPred={row["label_pred_des"]}'
        

        # title = f"True: {row['true_label_des']}\ngt({row['label_gt']}): {row['label_gt_des']}\npred({row['label_pred']}): {row['label_pred_des']}\nprob: {row['max_prob']:.3f}"
        imgname= os.path.basename(row["imgpath"])
        # title = f"{imgname}\ngt({row['label_gt']}): {row['label_gt_des']}\npred({row['label_pred']}): {row['label_pred_des']}\nprob: {row['max_prob']:.3f}"
        ax = fig.add_subplot(nrows, ncols, cnt + 1)

        # if label_gt_des == 'traffic sign':
        #     label_gt_des = 'sign'
        # elif label_gt_des == 'traffic light':
        #     label_gt_des = 'light'

        # disable title
        ax.set_title(title, fontdict={'fontsize': 7, 'fontweight': 5})
        # change between cropped image or the whole image
        # plt.imshow(img_with_bbox, cmap="gray")
        plt.imshow(bbox_img, cmap="gray")
            

        

        cnt+=1
        # disable title

        # plt.title(title, fontdict={'fontsize': 4, 'fontweight': 4})
        plt.axis("off")
        # print('-----')

    
    plt.tight_layout()
    # print(f'# of pyplot figure stack: {plt.get_fignums()}')
    # plt.suptitle(main_title)
    figpath = f'{fig_filepath}_{start}-{end}.png'

    plt.savefig(figpath, dpi=dpi, bbox_inches='tight')
    print(f'Save figure to {figpath}')
    if show_img:
        plt.show()


def crop_bbox(xmin, ymin, xmax, ymax, img, img_shape, pad=0):
    img_xmax = img_shape[0]
    img_ymax = img_shape[1]
    xmin = int(xmin) - pad if int(xmin) - pad > 0 else 0
    ymin = int(ymin) - pad if int(ymin) - pad > 0 else 0
    xmax = int(xmax) + pad if int(xmin) + pad > 0 else img_xmax
    ymax = int(ymax) + pad if int(ymax) + pad > 0 else img_ymax

    # print(f'img.shape={img.shape}')
    if img.ndim == 3:
        cropped = img[xmin:xmax, ymin:ymax, :]
    elif img.ndim == 2:
        cropped = img[xmin:xmax, ymin:ymax]
    # print(f'cropped.shape={cropped.shape}')
    return cropped


# without true label
# def draw_bboxes(dataset, img, gt_bboxes, gt_labels, gt_labels_des, pred_labels, pred_labels_des, name2RGB, name2white, isshow, isfont):
#     for idx, (gt_bbox, gt_label, gt_label_des, pred_label, pred_label_des) in enumerate(zip(gt_bboxes, gt_labels, gt_labels_des, pred_labels, pred_labels_des)):

# with true label
def draw_bboxes(dataset, img, gt_bboxes, gt_labels, gt_labels_des, pred_labels, pred_labels_des, true_labels_des, name2RGB, name2red, isshow, isfont):
    for idx, (gt_bbox, gt_label, gt_label_des, pred_label, pred_label_des, true_label_des) in enumerate(zip(gt_bboxes, gt_labels, gt_labels_des, pred_labels, pred_labels_des, true_labels_des)):
        
        # if dataset == 'shift':
        #     # x_center_ratio, y_center_ratio, w_ratio, h_ratio
        #     ymin, xmin, ymax, xmax = [int(i) for i in gt_bbox]

        # elif dataset == 'coco':
        # # elif dataset == 'coco':
        #     # cy, cx, h, w = [int(i) for i in gt_bbox]
        #     # ymin = int(cy - h/2.0)
        #     # xmin = int(cx - w/2.0)
        #     # ymax = int(cy + h/2.0)
        #     # xmax = int(cx + w/2.0)
        #     ymin, xmin, ymax, xmax = [int(i) for i in gt_bbox]
        #     # xmin, ymin, xmax, ymax = [int(i) for i in gt_bbox]

        # elif dataset == 'bdd100k':
        #     ymin, xmin, ymax, xmax = [int(i) for i in gt_bbox]


        ymin, xmin, ymax, xmax = [int(i) for i in gt_bbox]
            
        
        # print(f'[def draw_bboxes] img.shape={img.shape}')
        # print(f'[def draw_bboxes] xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}')
        # start = (xmin, ymin)
        # end = (xmax, ymax)
        r = [xmin, xmax, xmax, xmin, xmin]
        c = [ymax, ymax, ymin, ymin, ymax]
        rr, cc = polygon_perimeter(r, c, img.shape)
        # rr, cc = polygon_perimeter(start, end=end, shape=img.shape)

        if dataset in ['shift', 'coco']:
            if true_label_des != gt_label_des:
                rgb = name2red[gt_label_des]
            else:
                rgb = name2RGB[gt_label_des]

        # if dataset in ['bdd100k_gdino']:
        #     if 

        else:
            assert dataset in ['bdd100k', 'bdd100k_gdino']
            if true_label_des == 'TRUE':
                # c_incorrect is true
                rgb = name2red[gt_label_des]
            else:
                # c_incorrect is false or Not Detected
                rgb = name2RGB[gt_label_des]


        if img.ndim == 3:
            img[rr, cc ,0] = rgb[0]
            img[rr, cc ,1] = rgb[1]
            img[rr, cc ,2] = rgb[2]

        elif img.ndim == 2:
            raise NotImplementedError
            # TODO buggy here
            # only use 1st channel
            img[rr, cc] = rgb[0]
            # img[rr, cc] = rgb[0]

        if isfont is True:
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_org = (ymin, xmin-2)

            if gt_label_des == 'traffic sign':
                gt_label_des = 'sign'
            elif gt_label_des == 'traffic light':
                gt_label_des = 'light'

            labeltext = f'{gt_label_des}'
            cv2.putText(img, labeltext, text_org, font, 0.5, rgb, 1)

    # print(f'img.shape={img.shape}')
    # img = np.moveaxis(img, 0, -1)
    
    # dpi = 150
    # fig = plt.figure(dpi=dpi, figsize=(12,14))
    # plt.imshow(img)
    if isshow is True:
        dpi = 150
        fig = plt.figure(dpi=dpi, figsize=(7,9))
        plt.imshow(img)
    
    return img


def crop_bboxes(img, img_shape, gt_bboxes, gt_labels, pad):
    bbox_imgs = []

    for idx, (bbox, labels) in enumerate(zip(gt_bboxes, gt_labels)):
        ymin, xmin, ymax, xmax = bbox
        
        bbox_img = crop_bbox(xmin, ymin, xmax, ymax, img, img_shape, pad)
        bbox_imgs.append(bbox_img)

    return bbox_imgs


def get_label_issues(bbox_gt_labels, bbox_probs, frac_noise, filter_by, ranked_by):

    issues = find_label_issues(
        labels=bbox_gt_labels,
        pred_probs=bbox_probs,
        frac_noise=frac_noise,
        filter_by=filter_by,
        return_indices_ranked_by=ranked_by,
        # num_to_remove_per_class=remove_per_class
    )

    return issues


def add_loss_ranking(df):

    if 'loss' not in df.columns:
        print(f'Add loss column to df')
        df = df.assign(loss = df['label_gt_prob'].apply(lambda x: -np.log(x)))

    loss_ranking_index = df['loss'].sort_values(ascending=False, inplace=False).index
    loss_ranking_mapper = {idx: ranking for ranking, idx in enumerate(loss_ranking_index)}

    df['LRank'] = df.index.to_series().map(loss_ranking_mapper)

    max_loss_idx = df['loss'].argmax()
    assert df['LRank'][max_loss_idx] == 0, 'max loss index is not 0'

    return df


def gen_one_hot(labels):
    n_values = np.max(labels) + 1
    return np.eye(n_values)[labels]

def get_confidence_for_second_label(bbox_gt_labels, bbox_probs):
    removed_bbox_probs = np.copy(bbox_probs)
    onehot_gt_labels = gen_one_hot(bbox_gt_labels)
    keep_bool = ~onehot_gt_labels.astype(bool)
    gt0_bbox_probs = np.where(keep_bool, removed_bbox_probs, 0)
    second_label_confidence = np.max(gt0_bbox_probs, axis=1)
    return second_label_confidence


def add_score(df, ranked_by, bbox_gt_labels, bbox_probs, cl_idxs):
    # normalized_margin, self_confidence, ratio_confidence, entropy, relative_entropy
    if ranked_by == 'normalized_margin':
        margins = get_normalized_margin_for_each_label(bbox_gt_labels, bbox_probs)
        df[shorten[ranked_by]] = margins

    elif ranked_by == 'self_confidence':
        self_confidences = get_self_confidence_for_each_label(bbox_gt_labels, bbox_probs)
        df[shorten[ranked_by]] = self_confidences

    elif ranked_by == 'relative_confidence':
        self_confidences = get_self_confidence_for_each_label(bbox_gt_labels, bbox_probs)
        second_label_confidence = get_confidence_for_second_label(bbox_gt_labels, bbox_probs)
        relative_confidences = second_label_confidence / self_confidences
        df[shorten[ranked_by]] = relative_confidences

    elif ranked_by == 'entropy':
        entropy = -np.sum(bbox_probs * np.log(bbox_probs), axis=1)
        df[shorten[ranked_by]] = entropy*1e4

    elif ranked_by == 'relative_entropy':
        onehot_gt_labels = gen_one_hot(bbox_gt_labels)
        epsilon = 1e-10
        onehot_gt_labels += epsilon
        kl_div = np.sum(bbox_probs * np.log(bbox_probs / onehot_gt_labels), axis=1)
        df[shorten[ranked_by]] = kl_div*1e4

    else:
        print(f'Wrong ranked_by={ranked_by}')
        return df

    if ranked_by in ['relative_confidence', 'entropy', 'relative_entropy']:
        new_colname = f'CL_{shorten[ranked_by]}'

        if ranked_by in ['entropy']:
            df.sort_values(by=shorten[ranked_by], ascending=True, inplace=True)

        elif ranked_by in ['relative_confidence', 'relative_entropy']:
            df.sort_values(by=shorten[ranked_by], ascending=False, inplace=True)

        ranked_indexes = df.index.to_list()

        not_detected_indexes = set(ranked_indexes) - set(cl_idxs)

        issue_dic = {idx: ranking for ranking, idx in  enumerate(ranked_indexes)}
        # for ranking, idx in enumerate(ranked_indexes ):
        #     issue_dic[idx] = ranking

        for idx in issue_dic.keys():
            if idx in not_detected_indexes:
                issue_dic[idx] = 999999
            assert issue_dic[idx] is not None

        series = pd.Series(issue_dic, index=issue_dic.keys())

        df[new_colname] = series

    return df


def add_CL_ranking(df, filter_by, bbox_gt_labels, bbox_probs):

    for ranked_by in ranked_by_options:
        
        frac_noise = 1.0
        label_issues = get_label_issues(bbox_gt_labels, bbox_probs, frac_noise, filter_by, ranked_by)
        bbox_indexes = df.index.to_list()
        

        # print(f'ranked_by={ranked_by}, label_issues(#={len(label_issues)})={label_issues[:5]}')

        key = f'{shorten[filter_by]}_{shorten[ranked_by]}'
        # print(f'Add {key} to df')

        label_issues_set = set(label_issues)

        not_detected_indexes = set(bbox_indexes) - label_issues_set

        issue_dic = {i: None for i in bbox_indexes}
        for ranking, idx in enumerate(label_issues):
            issue_dic[idx] = ranking

        for idx in issue_dic.keys():
            if idx in not_detected_indexes:
                issue_dic[idx] = 999999
            assert issue_dic[idx] is not None

        series = pd.Series(issue_dic, index=issue_dic.keys())

        df[key] = series
        
        # print('--------')
        
    return df

def run(df):

    ############################################################
    #                     Run CL methods                       #
    ############################################################
    # CSV columns, index is unique index for each cropped images
    # ,prob,max_prob,label_gt,label_gt_des,label_pred,label_pred_des,loss,imgpath,imgname,PBNR_NMargin,PBNR_SConf,PBNR_CWE,CL_NMargin,CL_SConf,CL_CWE,ArgMax_NMargin,ArgMax_SConf,ArgMax_CWE,LRank


    ###############################################################################
    #                              prune_by_noise_rate                            #
    ###############################################################################
    bbox_gt_labels = df['label_gt'].to_numpy()
    bbox_probs = np.stack(df['prob'].to_numpy(), axis=0)
    # df = add_CL_ranking(df, 'prune_by_noise_rate', bbox_gt_labels, bbox_probs)
    df = add_CL_ranking(df, 'confident_learning', bbox_gt_labels, bbox_probs)
    # df = add_CL_ranking(df, 'predicted_neq_given', bbox_gt_labels, bbox_probs)
    df = add_loss_ranking(df)

    cl_idxs = df.loc[df['CL_SC'] != 999999].index.values

    df = add_score(df, 'normalized_margin', bbox_gt_labels, bbox_probs, cl_idxs)
    df = add_score(df, 'self_confidence', bbox_gt_labels, bbox_probs, cl_idxs)
    # df = add_score(df, 'relative_confidence', bbox_gt_labels, bbox_probs, cl_idxs)
    # df = add_score(df, 'entropy', bbox_gt_labels, bbox_probs, cl_idxs)
    # df = add_score(df, 'relative_entropy', bbox_gt_labels, bbox_probs, cl_idxs)

    # Ranked higher(idx=0) when value is lower
    assert df['NM'].argmin() == df['CL_NM'].argmin()

    # Not always
    # assert df['SC'].argmin() == df['CL_SC'].argmin()
    
    return df



def cal_acc(df, method):

    detected_items = df.loc[df[method] != 999999].copy()
    untouched_items = df.loc[df['label_gt'] == df['true_label']]
    relevant_items = df.loc[df['label_gt'] != df['true_label']]

    corrupted_percentage = 100* len(relevant_items) / len(df)

    detected_items.sort_values(method, ascending=True, inplace=True)

    # detected_items['TP_all'] = detected_items['label_gt'] != detected_items['true_label']
    detected_items = detected_items.assign(TP_all = detected_items['label_gt'] != detected_items['true_label'])

    TP_all = detected_items['TP_all'].value_counts()[True]
    precision = 100 * TP_all / len(detected_items)
    recall = 100 * TP_all / len(relevant_items)

    print(f'All syntastic label errors={len(relevant_items)}, corrupted_percentage={corrupted_percentage:.3f}%')
    print(f'{method} precision = {precision}%, recall={recall}%')

    # Top 50% 
    top50_detected_items = detected_items[:int(len(detected_items)*0.5)]
    top50_detected_items.loc[top50_detected_items['label_gt'] != top50_detected_items['true_label'], 'TP'] = top50_detected_items['label_gt'] != top50_detected_items['true_label']
    # top50_detected_items['TP'] = top50_detected_items['label_gt'] != top50_detected_items['true_label']
    TP50 = top50_detected_items['TP'].value_counts()[True]
    tp50_precision = 100 * TP50 / len(top50_detected_items)
    print(f'Top 50% precision={tp50_precision}')

    # Top 20% 
    top20_detected_items = detected_items[:int(len(detected_items)*0.2)]
    top20_detected_items['TP'] = top20_detected_items['label_gt'] != top20_detected_items['true_label']
    TP20 = top20_detected_items['TP'].value_counts()[True]
    tp20_precision = 100 * TP20 / len(top20_detected_items)
    print(f'Top 20% precision={tp20_precision}')

    # Top 10% 
    top10_detected_items = detected_items[:int(len(detected_items)*0.1)]
    top10_detected_items['TP'] = top10_detected_items['label_gt'] != top10_detected_items['true_label']
    TP10 = top10_detected_items['TP'].value_counts()[True]
    tp10_precision = 100 * TP10 / len(top10_detected_items)
    print(f'Top 10% precision={tp10_precision}')

    print('==========================================')



def convert_csv2np(df, colnames):

    for col in colnames:
        # print(f'Convert {col} to np array')
        df[col] = df[col].apply(lambda x: np.asarray(x.split(';'), dtype=float))

    return df



def convert_np2csv(df, colnames):
    for col in colnames:

        if col == 'bbox_gt':
            # print(f'Convert {col} np array to csv string')
             df[col] = df[col].apply(lambda x: ';'.join([f'{int(p)}' for p in x]).replace(' ', ''))
        else:
            df[col] = df[col].apply(lambda x: ';'.join([f'{p:.10f}' for p in x]).replace(' ', ''))

    return df


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


def gen_shift_mapper(shift_subset):
    shift_all = {
        0: "pedestrian",
        1: "car",
        2: "truck",
        3: "bus",
        4: "motorcycle",
        5: "bicycle"
    }
    mapper = gen_id_mapper(shift_all, shift_subset)
    return mapper

def gen_coco_mapper(det_all, subset):
    mapper = gen_id_mapper(det_all, subset)
    return mapper


def process_shift_gdino(dataset, m_name, subset):
    from id_to_class import shift_det as id2class
    from id_to_class import shift_det_name2red as name2red
    from id_to_class import shift_det_name2white as name2white

    if dataset == 'shift':
        from id_to_class import shift_det as id2class
        if 6 in id2class.keys():
            del id2class[6]

        if subset == 'all':
            shift_subset = {
                0: "pedestrian",
                1: "car",
                2: "truck",
                3: "bus",
                4: "motorcycle",
                5: "bicycle"
            }
        elif subset == 'vehicle':
            shift_subset = {
                0: "car",
                1: "truck",
                2: "bus",
                # 3: "pedestrian",
                # 4: "motorcycle",
                # 5: "bicycle"
            }

        elif subset == 'moto' or subset == 'mobike':
            shift_subset = {
                0: "pedestrian",
                1: "motorcycle",
                2: "bicycle"
                # 3: "car",
                # 4: "truck",
                # 5: "bus",
            }
        elif subset == 'heter':
            shift_subset = {
                # 3: "pedestrian",
                0: "motorcycle",
                # 4: "bicycle"
                1: "car",
                # 5: "truck",
                2: "bus",
            }
        elif subset == 'carped':
            shift_subset = {
                0: "car",
                1: "pedestrian",
                # 2: "truck",
                # 3: "bus",
                # 4: "motorcycle",
                # 5: "bicycle"
            }
        elif subset == 'carmo':
            shift_subset = {
                0: "car",
                1: "motorcycle",
                # 2: "truck",
                # 3: "bus",
                # 4: "pedestrian",
                # 5: "bicycle"
            }
        else:
            print(f'Wrong subset name={subset}')
            return

        id2class_subset = shift_subset
        id2new = gen_shift_mapper(shift_subset)

        # multiple files
        main_dirname = 'SHIFT_gdino_swint'
        source_csv_dirpath = f'./postprocessing/code/data_pred/{main_dirname}/{m_name}'
        target_csv_dirpath = f'./postprocessing/code/DataFrame/{main_dirname}/{m_name}'

        # single file
        # fig_dirpath = '/home/belay/Documents/0_UPPSALA/KOGNIC_research/Codes/trafficsign/fig/SHIFT_gdino'
        # main_dirname = 'SHIFT_gdino_swint'
        # source_csv_dirpath = f'./postprocessing/code/data_pred/{main_dirname}/'
        # source_csv_filename = f'val_{m_name}_box0.2_{subset}.csv'
        # source_csv_filepath = f'{source_csv_dirpath}/{source_csv_filename}'
        # target_csv_dirpath = f'./postprocessing/code/DataFrame/{main_dirname}'
    
    rounds = range(1 , 6)
    # rounds = [1]
    for round_i in rounds:
        # multiple files
        fig_dirpath = f'/home/belay/Documents/0_UPPSALA/KOGNIC_research/Codes/trafficsign/fig/{main_dirname}/{m_name}'
        source_csv_filename = f'{m_name}_round{round_i}.csv'
        source_csv_filepath = f'{source_csv_dirpath}/{source_csv_filename}'
        target_csv_filepath = f'{target_csv_dirpath}/{source_csv_filename}'

        if not os.path.exists(target_csv_dirpath):
            os.makedirs(target_csv_dirpath)


        dino_df = pd.read_csv(source_csv_filepath, header=0, delimiter=',', index_col=0)

        colnames = ['bbox_gt', 'class_logit', 'class_logit_raw', 'prob', 'prob_sig']
        dino_df = convert_csv2np(dino_df, colnames)

        # use prob before sigmoid
        bbox_probs = np.stack(dino_df['prob'].to_numpy(), axis=0)
        bbox_probs_sig = np.stack(dino_df['prob_sig'].to_numpy(), axis=0)
        bbox_gt_labels = dino_df['label_gt'].to_numpy()

        label_gt_prob = bbox_probs[np.arange(len(bbox_gt_labels)), bbox_gt_labels]
        label_gt_prob_sig = bbox_probs_sig[np.arange(len(bbox_gt_labels)), bbox_gt_labels]
        dino_df['label_gt_prob'] = label_gt_prob
        dino_df['label_gt_prob_sig'] = label_gt_prob_sig

        # Slow
        # def get_gt_prob(row):
        #     label_id = row['label_gt']
        #     label_gt_prob = row['prob'][label_id]
        #     label_gt_prob_sig = row['prob_sig'][label_id]
        #     return pd.Series([label_gt_prob, label_gt_prob_sig])
        # Too slow
        # dino_df[['label_gt_prob', 'label_gt_prob_sig']] = dino_df.apply(get_gt_prob ,axis=1)
        # dino_df['label_gt_prob'] = dino_df['prob'].apply(lambda x: id2class_subset[x])

        # Overwrite max_prob with label_gt_prob
        # Overwrite max_prob_sig with label_gt_prob_sig
        # dino_df.drop(labels='max_prob', axis=1, inplace=True)
        # dino_df.drop(labels='max_prob_sig', axis=1, inplace=True)



        dino_df['true_label'] = dino_df['true_label'].apply(lambda x: id2new[x])

        # Overwrite label_pred_des
        dino_df['label_pred_des'] = dino_df['label_pred'].apply(lambda x: id2class_subset[x])

        
        # use prob after sigmoid
        # bbox_probs = np.stack(dino_df['prob_sig'].to_numpy(), axis=0)

        # dino_df = add_CL_ranking(dino_df, 'both', bbox_gt_labels, bbox_probs)
        # dino_df = add_CL_ranking(dino_df, 'prune_by_class', bbox_gt_labels, bbox_probs)
        # dino_df = add_CL_ranking(dino_df, 'prune_by_noise_rate', bbox_gt_labels, bbox_probs)
        dino_df = add_CL_ranking(dino_df, 'confident_learning', bbox_gt_labels, bbox_probs)
        # dino_df = add_CL_ranking(dino_df, 'predicted_neq_given', bbox_gt_labels, bbox_probs)
        dino_df = add_loss_ranking(dino_df)

        cl_idxs = dino_df.loc[dino_df['CL_SC'] != 999999].index.values

        dino_df = add_score(dino_df, 'normalized_margin', bbox_gt_labels, bbox_probs, cl_idxs)
        dino_df = add_score(dino_df, 'self_confidence', bbox_gt_labels, bbox_probs, cl_idxs)
        # dino_df = add_score(dino_df, 'relative_confidence', bbox_gt_labels, bbox_probs, cl_idxs)
        # dino_df = add_score(dino_df, 'entropy', bbox_gt_labels, bbox_probs, cl_idxs)
        # dino_df = add_score(dino_df, 'relative_entropy', bbox_gt_labels, bbox_probs, cl_idxs)

        # Ranked higher(idx=0) when value is lower
        assert dino_df['NM'].argmin() == dino_df['CL_NM'].argmin()
        assert dino_df['SC'].argmin() == dino_df['CL_SC'].argmin()

        img = np.array(Image.open(dino_df['imgpath'][0]))
        img_w, img_h = img.shape[1], img.shape[0]
        print(f'img_w={img_w}, img_h={img_h}')
        dino_df['bbox_gt'] = dino_df['bbox_gt'].apply(lambda x: ratio_cxcywh2xyxy(img_w, img_h, x))

        ######## Draw figures ########
        # methods = ['NM', 'SC', 'CL_NM', 'CL_SC']
        methods = ['SC']
        for method in methods:
            if not os.path.exists(fig_dirpath):
                os.makedirs(fig_dirpath)

            fig_filepath = f'{fig_dirpath}/r{round_i}_{method}.png'

            # cal_acc(dino_df, method)
            plot_issues(dino_df, dataset, method, name2white, name2red, fig_filepath, show_img=False)

        print(f'dino_df={dino_df.head()}')
        ##########################

        dino_df = convert_np2csv(dino_df, colnames)

        dino_df.to_csv(f'{target_csv_filepath}', index=True, header=True, float_format='%.5f')
        print(f'Saved to {target_csv_filepath}')





def process_coco_gdino(dataset, m_name, subset):

    if dataset == 'coco':
        from id_to_class import coco_shiftall_det as id2class
        from id_to_class import coco_shiftall_name2red as name2red
        from id_to_class import coco_shiftall_name2white as name2white

        if subset == 'shiftall':
            coco_subset = {
                0: "car",
                1: "truck",
                2: "bus",
                3: "person",
                4: "motorcycle",
                5: "bicycle"
            }
        elif subset == 'vehicle':
            coco_subset = {
                0: "car",
                1: "truck",
                2: "bus"
            }

        elif subset == 'moto':
            coco_subset = {
                0: "person",
                1: "motorcycle",
                2: "bicycle"
            }
        else:
            print(f'Wrong subset name={subset}')
            return

        id2class_subset = coco_subset
        # id2new = gen_coco_mapper(id2class, coco_subset)

        main_dirname = 'COCO_gdino_swinb'
        # fig_dirpath = f'/home/belay/Documents/0_UPPSALA/KOGNIC_research/Codes/trafficsign/fig/{main_dirname}'
        # annotations_instances_val2017_box0.2_all.csv
        # m_name = 'instances_val2017'

        # single file
        # source_csv_dirpath = f'./postprocessing/code/data_pred/{main_dirname}'
        # source_csv_filename = f'{subset}_val2017_{m_name}_box0.2_{subset}.csv'
        # source_csv_filepath = f'{source_csv_dirpath}/{source_csv_filename}'
        # target_csv_dirpath = f'./postprocessing/code/DataFrame/{main_dirname}'

        # multiple files
        source_csv_dirpath = f'./postprocessing/code/data_pred/{main_dirname}/{m_name}'
        target_csv_dirpath = f'./postprocessing/code/DataFrame/{main_dirname}/{m_name}'

        if not os.path.exists(target_csv_dirpath):
            os.makedirs(target_csv_dirpath)

    for round_i in range(1, 6):
        # multiple files
        fig_dirpath = f'/home/belay/Documents/0_UPPSALA/KOGNIC_research/Codes/trafficsign/fig/{main_dirname}/{m_name}/big'
        source_csv_filename = f'{subset}_{m_name}_round{round_i}.csv'
        source_csv_filepath = f'{source_csv_dirpath}/{source_csv_filename}'
        target_csv_filepath = f'{target_csv_dirpath}/{source_csv_filename}'

        dino_df = pd.read_csv(source_csv_filepath, header=0, delimiter=',', index_col=0)

        colnames = ['bbox_gt', 'class_logit', 'class_logit_raw', 'prob', 'prob_sig']
        dino_df = convert_csv2np(dino_df, colnames)

        # use prob before sigmoid
        bbox_probs = np.stack(dino_df['prob'].to_numpy(), axis=0)
        bbox_probs_sig = np.stack(dino_df['prob_sig'].to_numpy(), axis=0)
        bbox_gt_labels = dino_df['label_gt'].to_numpy()

        label_gt_prob = bbox_probs[np.arange(len(bbox_gt_labels)), bbox_gt_labels]
        label_gt_prob_sig = bbox_probs_sig[np.arange(len(bbox_gt_labels)), bbox_gt_labels]
        dino_df['label_gt_prob'] = label_gt_prob
        dino_df['label_gt_prob_sig'] = label_gt_prob_sig

        # Slow
        # def get_gt_prob(row):
        #     label_id = row['label_gt']
        #     label_gt_prob = row['prob'][label_id]
        #     label_gt_prob_sig = row['prob_sig'][label_id]
        #     return pd.Series([label_gt_prob, label_gt_prob_sig])
        # Too slow
        # dino_df[['label_gt_prob', 'label_gt_prob_sig']] = dino_df.apply(get_gt_prob ,axis=1)
        # dino_df['label_gt_prob'] = dino_df['prob'].apply(lambda x: id2class_subset[x])

        # Overwrite max_prob with label_gt_prob
        # Overwrite max_prob_sig with label_gt_prob_sig
        # dino_df.drop(labels='max_prob', axis=1, inplace=True)
        # dino_df.drop(labels='max_prob_sig', axis=1, inplace=True)



        # dino_df['true_label'] = dino_df['true_label'].apply(lambda x: id2new[x])

        # Overwrite label_pred_des
        dino_df['label_pred_des'] = dino_df['label_pred'].apply(lambda x: id2class_subset[x])

        
        # use prob after sigmoid
        # bbox_probs = np.stack(dino_df['prob_sig'].to_numpy(), axis=0)

        # dino_df = add_CL_ranking(dino_df, 'both', bbox_gt_labels, bbox_probs)
        # dino_df = add_CL_ranking(dino_df, 'prune_by_class', bbox_gt_labels, bbox_probs)
        # dino_df = add_CL_ranking(dino_df, 'prune_by_noise_rate', bbox_gt_labels, bbox_probs)
        dino_df = add_CL_ranking(dino_df, 'confident_learning', bbox_gt_labels, bbox_probs)
        # dino_df = add_CL_ranking(dino_df, 'predicted_neq_given', bbox_gt_labels, bbox_probs)
        dino_df = add_loss_ranking(dino_df)

        cl_idxs = dino_df.loc[dino_df['CL_SC'] != 999999].index.values

        dino_df = add_score(dino_df, 'normalized_margin', bbox_gt_labels, bbox_probs, cl_idxs)
        dino_df = add_score(dino_df, 'self_confidence', bbox_gt_labels, bbox_probs, cl_idxs)
        # dino_df = add_score(dino_df, 'relative_confidence', bbox_gt_labels, bbox_probs, cl_idxs)
        # dino_df = add_score(dino_df, 'entropy', bbox_gt_labels, bbox_probs, cl_idxs)
        # dino_df = add_score(dino_df, 'relative_entropy', bbox_gt_labels, bbox_probs, cl_idxs)

        # Ranked higher(idx=0) when value is lower
        assert dino_df['NM'].argmin() == dino_df['CL_NM'].argmin()
        # assert dino_df['SC'].argmin() == dino_df['CL_SC'].argmin()

        # BUG!!! not all image has the same size
        # img = np.array(Image.open(dino_df['imgpath'][0]))
        # img_w, img_h = img.shape[1], img.shape[0]

        dino_df['img_w']  = dino_df['imgpath'].apply(lambda x: np.array(Image.open(x)).shape[1])
        dino_df['img_h']  = dino_df['imgpath'].apply(lambda x: np.array(Image.open(x)).shape[0])

        # dino_df['bbox_gt'] = dino_df['bbox_gt'].apply(lambda x: ratio_cxcywh2xyxy(img_w, img_h, x))
        dino_df['bbox_gt'] = dino_df.apply(lambda x: ratio_cxcywh2xyxy(x['img_w'], x['img_h'], x['bbox_gt']), axis=1)

        ######## Draw figures ########
        # methods = ['CL_NM', 'CL_SC']
        methods = ['NM', 'SC']
        for method in methods:
            if not os.path.exists(fig_dirpath):
                os.makedirs(fig_dirpath)

            fig_filepath = f'{fig_dirpath}/r{round_i}_{method}.png'

            cal_acc(dino_df, method)
            plot_issues(dino_df, dataset, method, name2white, name2red, fig_filepath, show_img=False)

        print(f'dino_df={dino_df.head()}')
        ##########################

        # dino_df = convert_np2csv(dino_df, colnames)

        # dino_df.to_csv(f'{target_csv_filepath}', index=True, header=True, float_format='%.5f')
        # print(f'Saved to {target_csv_filepath}')


def process_shift_fasterrcnn(input_m_name, keep_bg):
    from id_to_class import shift_det as id2class
    from id_to_class import shift_det_name2RGB as name2RGB
    from id_to_class import shift_det_name2white as name2white
    from id_to_class import shift_det_name2red as name2red

    if 6 in id2class.keys():
        del id2class[6]

    dataset = 'shift'
    # epos = range(1, 9)
    epos = [8]


    if input_m_name == 'clean':
        m_name = f'faster_rcnn_r50_fpn_1x_det_SHIFT'
    elif input_m_name[:6] == 'clean_':
        m_name =  input_m_name
    else:
        m_name = f'faster_rcnn_r50_fpn_1x_det_SHIFT_{input_m_name}'

    source_csv_dirpath = f'./postprocessing/code/data_pred/SHIFT_val/{m_name}'
    target_csv_dirpath = f'./postprocessing/code/DataFrame/SHIFT_val/{m_name}'

    fig_dirpath = f'/home/belay/Documents/0_UPPSALA/KOGNIC_research/Codes/trafficsign/fig/SHIFT_rcnn/{input_m_name}'

    if not os.path.isdir(fig_dirpath):
        print(f'Create fig_dirpath: {fig_dirpath}')
        os.makedirs(fig_dirpath)

    for epo in epos:


        source_csv_path = f'{source_csv_dirpath}/epo{epo}.csv'


        start_time = time.time()

        # colnames = ['imgpath','cls_score','prob','max_prob','bbox_gt','true_label','true_label_des','label_gt','label_gt_des','label_pred','label_pred_des','loss']
        epo_df = pd.read_csv(source_csv_path, header=0, delimiter=',', index_col=0)
        colnames = ['bbox_gt', 'prob', 'cls_score']
        epo_df = convert_csv2np(epo_df, colnames)

        if keep_bg is False:
            # Remove last bg probability
            assert epo_df['cls_score'][0].shape[0] == 7, 'cls_score does not have 7 classes'

            # Remove last bg probability
            epo_df['cls_score'] = epo_df['cls_score'].apply(lambda x: x[:-1])
            epo_df['prob'] = epo_df['cls_score'].apply(lambda x: cls_to_probs(x))
            epo_df['max_prob'] = epo_df['prob'].apply(lambda x: x.max())
            epo_df['label_pred'] = epo_df['prob'].apply(lambda x: x.argmax())

            epo_df['label_pred_des'] = epo_df['label_pred'].apply(lambda x: id2class[int(x)])

            epo_df['loss'] = epo_df[['prob', 'label_gt']].apply(lambda x: -np.log(x[0][x[1]]), axis=1)

        

        epo_df = run(epo_df)

        methods = ['SC','CL_SC', 'NM', 'CL_NM']
        for method in methods:

            fig_filepath = f'{fig_dirpath}/{input_m_name}_epo{epo}_{method}'
            # cal_acc(epo_df, method)
            plot_issues(epo_df, dataset, method, name2white, name2red, fig_filepath, show_img=False)

        # if keep_bg is True:
        #     target_csv_dirpath = f'./postprocessing/code/DataFrame/SHIFT_fasterrcnn_bg/{input_m_name}'
        # else:
        #     target_csv_dirpath = f'./postprocessing/code/DataFrame/SHIFT/{input_m_name}'

        # if not os.path.isdir(target_csv_dirpath):
        #     print(f'Create csv_dirpath: {target_csv_dirpath}')
        #     os.mkdir(target_csv_dirpath)

        # epo_df = convert_np2csv(epo_df, colnames)
        # epo_df.to_csv(f'{target_csv_dirpath}/epo{epo}.csv', index=True, header=True, float_format='%.5f')
        # print(f'Saved to {target_csv_dirpath}/epo{epo}.csv')

        # print("--- %s seconds ---" % (time.time() - start_time))


        gc.collect()
        print('------------------------\n')


def save_bdd100k_fasterrcnn(dataset):

    # epos = range(1, 13)
    # epos = [1]

    if dataset == 'bdd100k':

        from id_to_class import bdd100k_det as id2class
        if 6 in id2class.keys():
            del id2class[6]

        # m_name = 'bdd100k_fasterrcnn'
        m_name = 'bdd100k_gdino_swint'
        # m_name = 'bdd100k_gdino_swinb'

        fig_dirpath = '/home/belay/Documents/0_UPPSALA/KOGNIC_research/Codes/trafficsign/fig/BDD100k_Det3'
        source_csv_dirpath = f'./postprocessing/code/data_pred/{m_name}/val_all'

    elif dataset == 'shift':

        from id_to_class import shift_det as id2class
        fig_dirpath = '/home/belay/Documents/0_UPPSALA/KOGNIC_research/Codes/trafficsign/fig/SHIFT'
        source_csv_dirpath = f'./postprocessing/code/DataFrame/SHIFT/clean'

    source_csv_path = f'{source_csv_dirpath}/epo8.csv'
    df = pd.read_csv(source_csv_path, header=0, delimiter=',', index_col=0)
    colnames = ['bbox_gt', 'prob', 'cls_score']
    df = convert_csv2np(df, colnames)

    print(f'id2class={id2class}')
    for classname in id2class.values():
        print(f'classname={classname}')
        if classname != 'pedestrian':
            continue
        # df sort ascending=False --> No label error
        # df = df.sort_values('CL_NMargin', ascending=False, inplace=False)[:topn]

        cnt = 0
        cls_df = df.loc[df['label_gt_des'] == classname, ['label_gt', 'label_gt_des', 'imgpath', 'bbox_gt', 'label_pred', 'label_pred_des', 'max_prob']][150:]
        # print(f'cls_df={cls_df.head(20)}')

        for bbox_idx, row in cls_df.iterrows():
            img = np.array(Image.open(row['imgpath']))
            img_shape = img.shape

            isshow = False
            isfont = False
            img_with_bbox = draw_bboxes(img, [row['bbox_gt']], [row['label_gt']], [row['label_gt_des']], [row['label_pred']], [row['label_pred_des']], name2RGB, name2white, isshow, isfont)

            bbox_img = crop_bboxes(img_with_bbox, img_shape, [row['bbox_gt']], [row['label_gt']], pad=15)[0]

            plt.imshow(bbox_img, cmap="gray")
            plt.axis('off')
            plt.savefig(f"{fig_dirpath}/{classname}_{cnt}.png", bbox_inches='tight')
            

            # print(f'bbox_idx={bbox_idx}, classname={classname}')
            cnt += 1

            if cnt > 40:
                break


def preprocess_bdd100k_fasterrcnn():

    epos = range(1, 13)
    # epos = [1]
    m_name = 'bdd100k_swin-t_val_split1'
    fig_dirpath = '/home/belay/Documents/0_UPPSALA/KOGNIC_research/Codes/trafficsign/fig/BDD100k_Det3'
    source_csv_dirpath = f'./postprocessing/code/data_pred/{m_name}'
    target_csv_dirpath = f'./postprocessing/code/data_pred/tmp_{m_name}'

    if not os.path.isdir(target_csv_dirpath):
        print(f'Create csv_dirpath: {target_csv_dirpath}')
        os.mkdir(target_csv_dirpath)


    for epo in epos:

        source_csv_path = f'{source_csv_dirpath}/epo{epo}.csv'

        with open(source_csv_path, 'r') as f:
            all_of_it = f.read()
            all_of_it = all_of_it.replace('true_label,true_label_des,', '')

        target_csv_path = f'{target_csv_dirpath}/epo{epo}.csv'
        with open(target_csv_path, 'w') as f:
            f.write(all_of_it)


def cls_to_probs(cls_scores):

    probs = softmax(cls_scores, axis=0)

    # sanity check
    # close_to_1 = probs.sum(axis=1)
    # print(f'close_to_1={close_to_1}')

    return probs


def process_bdd100k_fasterrcnn_combine_splits(keep_bg):
    from id_to_class import bdd100k_det as id2class
    from id_to_class import bdd100k_det_name2RGB as name2RGB
    from id_to_class import bdd100k_det_name2white as name2white
    from id_to_class import bdd100k_det_name2red as name2red

    epos = range(1, 13)
    # epos = [12]
    # m_names = ['swin-t_val_split0', 'swin-t_val_split1']
    m_names = ['val_split0', 'val_split1']

    fig_dirpath = '/home/belay/Documents/0_UPPSALA/KOGNIC_research/Codes/trafficsign/fig/BDD100K_rcnn/multiplot'

    if not os.path.isdir(fig_dirpath):
        print(f'Create fig_dirpath: {fig_dirpath}')
        os.makedirs(fig_dirpath)
    
    e_dic = {int(epo): None for epo in epos}

    start_time = time.time()


    for epo in epos:


        m_dic_per_epo = {m_name: None for m_name in m_names}

        for m_name in m_names:
            source_csv_dirpath = f'./postprocessing/code/data_pred/bdd100k_fasterrcnn/{m_name}'

            source_csv_path = f'{source_csv_dirpath}/epo{epo}.csv'

            # colnames = ['imgpath','cls_score','prob','max_prob','bbox_gt','true_label','true_label_des','label_gt','label_gt_des','label_pred','label_pred_des','loss']
            epo_df = pd.read_csv(source_csv_path, header=0, delimiter=',', index_col=0)
            colnames = ['bbox_gt', 'prob', 'cls_score']
            epo_df = convert_csv2np(epo_df, colnames)

            if keep_bg is False:
                # Remove last bg probability
                assert epo_df['cls_score'][0].shape[0] == 11, 'cls_score does not have 11 classes'

                # Remove last bg probability
                epo_df['cls_score'] = epo_df['cls_score'].apply(lambda x: x[:-1])
                epo_df['prob'] = epo_df['cls_score'].apply(lambda x: cls_to_probs(x))
                epo_df['max_prob'] = epo_df['prob'].apply(lambda x: x.max())
                epo_df['label_pred'] = epo_df['prob'].apply(lambda x: x.argmax())

                epo_df['label_pred_des'] = epo_df['label_pred'].apply(lambda x: id2class[int(x)])


                epo_df['loss'] = epo_df[['prob', 'label_gt']].apply(lambda x: -np.log(x[0][x[1]]), axis=1)

            m_dic_per_epo[m_name] = epo_df
            gc.collect()

        combined_df = pd.concat(m_dic_per_epo.values(), ignore_index=True)

        e_dic[epo] = combined_df


        epo_df = run(combined_df)

        ###############################################
        #                                    Draw multiple figures
        ###############################################
        default_csv_filepath = './postprocessing/code/DataFrame/BDD100K/FasterRCNN/default_val_s01.csv'
        default_df = pd.read_csv(default_csv_filepath, header=0, delimiter=',', index_col=0)
        merge_df = pd.merge(epo_df, default_df, left_index=True, right_index=True)

        dataset = 'bdd100k'
        methods = ['CL_SC', 'SC', 'CL_NM', 'NM']
        for method in methods:
            # cal_acc(epo_df, method)
            fig_filepath = f'{fig_dirpath}/split01_epo{epo}_{method}'
            plot_issues(merge_df, dataset, method, name2white, name2red, fig_filepath, show_img=False)

        # Hard-code the target name
        if keep_bg is True:

            if 'swin-t' in m_names[0]:
                target_csv_dirpath = f'./postprocessing/code/DataFrame/BDD100K/swin-t_val_split01_bg'
            else:
                target_csv_dirpath = f'./postprocessing/code/DataFrame/BDD100K/val_split01_bg'
        else:
            if 'swin-t' in m_names[0]:
                target_csv_dirpath = f'./postprocessing/code/DataFrame/BDD100K/swin-t_val_split01'
            else:
                target_csv_dirpath = f'./postprocessing/code/DataFrame/BDD100K/val_split01'

        if not os.path.isdir(target_csv_dirpath):
            print(f'Create csv_dirpath: {target_csv_dirpath}')
            os.mkdir(target_csv_dirpath)

        epo_df = convert_np2csv(epo_df, colnames)
        epo_df.to_csv(f'{target_csv_dirpath}/epo{epo}.csv', index=True, header=True)
        print(f'Saved to {target_csv_dirpath}/epo{epo}.csv')

        print("--- %s seconds ---" % (time.time() - start_time))
        print('------------------------\n')


def process_bdd100k_gdino_combine_splits(keep_bg):
    from id_to_class import bdd100k_det as id2class
    from id_to_class import bdd100k_det_name2RGB as name2RGB
    from id_to_class import bdd100k_det_name2white as name2white
    from id_to_class import bdd100k_det_name2red as name2red


    dataset = 'bdd100k_gdino'
    m_names = ['Split-0', 'Split-1']
    dirname = 'bdd100k_gdino_swint'

    fig_dirpath = f'/home/belay/Documents/0_UPPSALA/KOGNIC_research/Codes/trafficsign/fig/{dirname}/multiplot'

    target_csv_dirpath = f'./postprocessing/code/DataFrame/{dirname}'
    target_csv_filepath = f'{target_csv_dirpath}/split01.csv'

    if not os.path.isdir(fig_dirpath):
        print(f'Create fig_dirpath: {fig_dirpath}')
        os.makedirs(fig_dirpath)

    if not os.path.isdir(target_csv_dirpath):
        print(f'Create csv_dirpath: {target_csv_dirpath}')
        os.mkdir(target_csv_dirpath)
    
    m_dic = {m: None for m in m_names}

    start_time = time.time()

    # m_dic_per_epo = {m_name: None for m_name in m_names}

    for m_name in m_names:
        source_csv_dirpath = f'./postprocessing/code/data_pred/{dirname}'

        source_csv_filepath = f'{source_csv_dirpath}/{m_name}_box0.2_all.csv'

        dino_df = pd.read_csv(source_csv_filepath, header=0, delimiter=',', index_col=0)

        colnames = ['bbox_gt', 'class_logit', 'class_logit_raw', 'prob', 'prob_sig']
        dino_df = convert_csv2np(dino_df, colnames)

        dino_df['img_w']  = dino_df['imgpath'].apply(lambda x: np.array(Image.open(x)).shape[1])
        dino_df['img_h']  = dino_df['imgpath'].apply(lambda x: np.array(Image.open(x)).shape[0])

        # dino_df['bbox_gt'] = dino_df['bbox_gt'].apply(lambda x: ratio_cxcywh2xyxy(img_w, img_h, x))
        dino_df['bbox_gt'] = dino_df.apply(lambda x: ratio_cxcywh2xyxy(x['img_w'], x['img_h'], x['bbox_gt']), axis=1)

        m_dic[m_name] = dino_df
        gc.collect()

    combined_df = pd.concat(m_dic.values(), ignore_index=True)


    # use prob before sigmoid
    bbox_probs = np.stack(combined_df['prob'].to_numpy(), axis=0)
    bbox_probs_sig = np.stack(combined_df['prob_sig'].to_numpy(), axis=0)
    bbox_gt_labels = combined_df['label_gt'].to_numpy()

    label_gt_prob = bbox_probs[np.arange(len(bbox_gt_labels)), bbox_gt_labels]
    label_gt_prob_sig = bbox_probs_sig[np.arange(len(bbox_gt_labels)), bbox_gt_labels]

    combined_df['label_gt_prob'] = label_gt_prob
    combined_df['label_gt_prob_sig'] = label_gt_prob_sig

    print('')


    # Overwrite max_prob with label_gt_prob
    # Overwrite max_prob_sig with label_gt_prob_sig
    combined_df.drop(labels='max_prob', axis=1, inplace=True)
    combined_df.drop(labels='max_prob_sig', axis=1, inplace=True)

    combined_df = add_CL_ranking(combined_df, 'confident_learning', bbox_gt_labels, bbox_probs)
    cl_idxs = combined_df.loc[combined_df['CL_SC'] != 999999].index.values

    combined_df = add_score(combined_df, 'normalized_margin', bbox_gt_labels, bbox_probs, cl_idxs)
    combined_df = add_score(combined_df, 'self_confidence', bbox_gt_labels, bbox_probs, cl_idxs)

    # Ranked higher(idx=0) when value is lower
    assert combined_df['NM'].argmin() == combined_df['CL_NM'].argmin()
    # assert combined_df['SC'].argmin() == combined_df['CL_SC'].argmin()



     ###############################################
    #                                    Draw multiple figures
    ###############################################
    default_csv_filepath = './postprocessing/code/DataFrame/BDD100K/FasterRCNN/default_val_s01.csv'
    default_df = pd.read_csv(default_csv_filepath, header=0, delimiter=',', index_col=0)
    merge_df = pd.merge(combined_df, default_df, left_index=True, right_index=True)
    methods = ['NM', 'SC', 'CL_NM', 'CL_SC']
    # methods = ['SC']
    for method in methods:
        if not os.path.exists(fig_dirpath):
            os.makedirs(fig_dirpath)

        fig_filepath = f'{fig_dirpath}/{method}.png'

        # cal_acc(dino_df, method)
        plot_issues(merge_df, dataset, method, name2RGB, name2red, fig_filepath, show_img=False)

    print(f'combined_df={combined_df.head()}')
    ##########################

    combined_df = convert_np2csv(combined_df, colnames)

    combined_df.to_csv(f'{target_csv_filepath}', index=True, header=True, float_format='%.5f')
    print(f'Saved to {target_csv_filepath}')


        # dino_df['true_label'] = dino_df['true_label'].apply(lambda x: id2new[x])

        # # Overwrite label_pred_des
        # dino_df['label_pred_des'] = dino_df['label_pred'].apply(lambda x: id2class_subset[x])

        
        # # use prob after sigmoid
        # # bbox_probs = np.stack(dino_df['prob_sig'].to_numpy(), axis=0)

        # # dino_df = add_CL_ranking(dino_df, 'both', bbox_gt_labels, bbox_probs)
        # # dino_df = add_CL_ranking(dino_df, 'prune_by_class', bbox_gt_labels, bbox_probs)
        # # dino_df = add_CL_ranking(dino_df, 'prune_by_noise_rate', bbox_gt_labels, bbox_probs)
        # dino_df = add_CL_ranking(dino_df, 'confident_learning', bbox_gt_labels, bbox_probs)
        # # dino_df = add_CL_ranking(dino_df, 'predicted_neq_given', bbox_gt_labels, bbox_probs)
        # dino_df = add_loss_ranking(dino_df)

        # cl_idxs = dino_df.loc[dino_df['CL_SC'] != 999999].index.values

        # dino_df = add_score(dino_df, 'normalized_margin', bbox_gt_labels, bbox_probs, cl_idxs)
        # dino_df = add_score(dino_df, 'self_confidence', bbox_gt_labels, bbox_probs, cl_idxs)
        # # dino_df = add_score(dino_df, 'relative_confidence', bbox_gt_labels, bbox_probs, cl_idxs)
        # # dino_df = add_score(dino_df, 'entropy', bbox_gt_labels, bbox_probs, cl_idxs)
        # # dino_df = add_score(dino_df, 'relative_entropy', bbox_gt_labels, bbox_probs, cl_idxs)

        # # Ranked higher(idx=0) when value is lower
        # assert dino_df['NM'].argmin() == dino_df['CL_NM'].argmin()
        # assert dino_df['SC'].argmin() == dino_df['CL_SC'].argmin()

        # img = np.array(Image.open(dino_df['imgpath'][0]))
        # img_w, img_h = img.shape[1], img.shape[0]
        # print(f'img_w={img_w}, img_h={img_h}')
        # dino_df['bbox_gt'] = dino_df['bbox_gt'].apply(lambda x: ratio_cxcywh2xyxy(img_w, img_h, x))

        # ######## Draw figures ########
        # # methods = ['NM', 'SC', 'CL_NM', 'CL_SC']
        # methods = ['SC']
        # for method in methods:
        #     if not os.path.exists(fig_dirpath):
        #         os.makedirs(fig_dirpath)

        #     fig_filepath = f'{fig_dirpath}/r{round_i}_{method}.png'

        #     # cal_acc(dino_df, method)
        #     plot_issues(dino_df, dataset, method, name2white, name2red, fig_filepath, show_img=False)

        # print(f'dino_df={dino_df.head()}')
        # ##########################

        # dino_df = convert_np2csv(dino_df, colnames)

        # dino_df.to_csv(f'{target_csv_filepath}', index=True, header=True, float_format='%.5f')
        # print(f'Saved to {target_csv_filepath}')

        ###############################################
        #                                    Draw multiple figures
        ###############################################
        # default_csv_filepath = './postprocessing/code/DataFrame/BDD100K/FasterRCNN/default_val_s01.csv'
        # default_df = pd.read_csv(default_csv_filepath, header=0, delimiter=',', index_col=0)
        # merge_df = pd.merge(epo_df, default_df, left_index=True, right_index=True)

        # dataset = 'bdd100k'
        # methods = ['CL_SC', 'SC', 'CL_NM', 'NM']
        # for method in methods:
        #     # cal_acc(epo_df, method)
        #     fig_filepath = f'{fig_dirpath}/split01_epo{epo}_{method}'
        #     plot_issues(merge_df, dataset, method, name2white, name2red, fig_filepath, show_img=False)

        # # Hard-code the target name
        # if keep_bg is True:

        #     if 'swin-t' in m_names[0]:
        #         target_csv_dirpath = f'./postprocessing/code/DataFrame/BDD100K/swin-t_val_split01_bg'
        #     else:
        #         target_csv_dirpath = f'./postprocessing/code/DataFrame/BDD100K/val_split01_bg'
        # else:
        #     if 'swin-t' in m_names[0]:
        #         target_csv_dirpath = f'./postprocessing/code/DataFrame/BDD100K/swin-t_val_split01'
        #     else:
        #         target_csv_dirpath = f'./postprocessing/code/DataFrame/BDD100K/val_split01'

        # if not os.path.isdir(target_csv_dirpath):
        #     print(f'Create csv_dirpath: {target_csv_dirpath}')
        #     os.mkdir(target_csv_dirpath)

        # epo_df = convert_np2csv(epo_df, colnames)
        # epo_df.to_csv(f'{target_csv_dirpath}/epo{epo}.csv', index=True, header=True)
        # print(f'Saved to {target_csv_dirpath}/epo{epo}.csv')

        # print("--- %s seconds ---" % (time.time() - start_time))
        # print('------------------------\n')


if __name__ == '__main__':
    import sys
    # save_bdd100k_fasterrcnn(sys.argv[1])
    # preprocess_bdd100k_fasterrcnn()
    # process_bdd100k_fasterrcnn_combine_splits(keep_bg=False)
    # process_bdd100k_gdino_combine_splits(keep_bg=False)

    # keep_bg=True is not implemented
    # process_shift_fasterrcnn(sys.argv[1], False)

    # dataset, m_name, subset = sys.argv[1], sys.argv[2], sys.argv[3]
    # process_shift_gdino(dataset, m_name, subset)
    # process_coco_gdino(dataset, m_name, subset)