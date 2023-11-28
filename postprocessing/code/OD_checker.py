import os
import csv
import sys
import time
import random
import traceback

import matplotlib
from mpl_interactions import ioff, panhandler, zoom_factory
matplotlib.use('TkAgg')

from matplotlib.widgets import TextBox

import numpy as np
import pandas as pd
from PIL import Image
import cv2
from skimage.draw import polygon_perimeter
import torch.nn.functional as F
import matplotlib.pyplot as plt

from cleanlab.classification import CleanLearning
from cleanlab.filter import find_label_issues, find_predicted_neq_given, find_label_issues_using_argmax_confusion_matrix
from cleanlab.dataset import health_summary
from cleanlab.count import get_confident_thresholds
from cleanlab.rank import get_label_quality_scores, get_normalized_margin_for_each_label, get_self_confidence_for_each_label, get_confidence_weighted_entropy_for_each_label
from cleanlab.internal.label_quality_utils import (
    _subtract_confident_thresholds,
    get_normalized_entropy,
)

from id_to_class import bdd100k_det as id2class
from id_to_class import bdd100k_det_name2RGB as name2RGB
from id_to_class import bdd100k_det_name2white as name2white
from id_to_class import bdd100k_det_name2red as  name2red




def compare_idxes(df, colnames, topn, classnames, by_class):
    
    all_topn_idxs = set()

    for colname in colnames:
        topn_idxs = []
        if by_class is True:
            for idx, classname in enumerate(classnames):
                class_df = df.loc[df['label_gt_des'] == classname]

                if colname == 'LRank':
                    detected_df = class_df['LRank'].sort_values(ascending=True, inplace=False)[:topn]
                    topn_idxs = detected_df.index.tolist()
                else:
                    detected_df = class_df.loc[class_df[colname]!=999999, :]
                    if len(detected_df) == 0:
                        continue
                    elif len(detected_df) < topn:
                        topn_idxs = detected_df.index.tolist()
                    else:
                        topn_idxs = detected_df[colname].sort_values(ascending=True, inplace=False)[:topn].index.tolist()


                all_topn_idxs.update(topn_idxs)

            
        else:

            if colname == 'LRank':
                detected_df = df['LRank'].sort_values(ascending=True, inplace=False)[:topn]
                topn_idxs = detected_df.index.tolist()

            else:
                detected_df = df.loc[df[colname]!=999999]

                if len(detected_df) == 0:
                    continue
                elif len(detected_df) < topn:
                    topn_idxs = detected_df.index.tolist()
                else:
                    topn_idxs = detected_df[:topn].index.tolist()
        
            all_topn_idxs.update(topn_idxs)

        print(f'{colname} Number of detected labels = {len(topn_idxs)}')

    return all_topn_idxs



def crop_bbox(xmin, ymin, xmax, ymax, img, img_shape, pad=0):
    img_xmax = img_shape[0]
    img_ymax = img_shape[1]
    xmin = int(xmin) - pad if int(xmin) - pad > 0 else 0
    ymin = int(ymin) - pad if int(ymin) - pad > 0 else 0
    xmax = int(xmax) + pad if int(xmin) + pad > 0 else img_xmax
    ymax = int(ymax) + pad if int(ymax) + pad > 0 else img_ymax

    # print(f'img.shape={img.shape}')
    cropped = img[xmin:xmax, ymin:ymax, :]
    # print(f'cropped.shape={cropped.shape}')
    return cropped


def crop_bboxes(img, img_shape, gt_bboxes, gt_labels, pad):
    bbox_imgs = []

    for idx, (bbox, labels) in enumerate(zip(gt_bboxes, gt_labels)):
        ymin, xmin, ymax, xmax = bbox
        
        bbox_img = crop_bbox(xmin, ymin, xmax, ymax, img, img_shape, pad)
        bbox_imgs.append(bbox_img)

    return bbox_imgs


def draw_bboxes(img, 
                gt_bboxes, 
                gt_labels, 
                gt_labels_des, 
                pred_labels, 
                pred_labels_des, 
                name2RGB, 
                name2white, 
                isshow):

    for idx, (gt_bbox, gt_label, gt_label_des, pred_label, pred_label_des) in enumerate(zip(gt_bboxes, gt_labels, gt_labels_des, pred_labels, pred_labels_des)):

        ymin, xmin, ymax, xmax = [int(i) for i in gt_bbox]
        
        # start = (xmin, ymin)
        # end = (xmax, ymax)
        r = [xmin, xmax, xmax, xmin, xmin]
        c = [ymax, ymax, ymin, ymin, ymax]
        rr, cc = polygon_perimeter(r, c, img.shape)
        # rr, cc = polygon_perimeter(start, end=end, shape=img.shape)
        rgb = name2RGB[gt_label_des]
        img[rr, cc ,0] = rgb[0]
        img[rr, cc ,1] = rgb[1]
        img[rr, cc ,2] = rgb[2]

        #############################################################
        #                   put text on image
        #############################################################
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # text_org = (ymin, xmin-2)

        # if gt_label_des == 'traffic sign':
        #     gt_label_des = 'sign'
        # elif gt_label_des == 'traffic light':
        #     gt_label_des = 'light'

        # labeltext = f'{gt_label_des}'
        # cv2.putText(img, labeltext, text_org, font, 0.5, rgb, 2)
        #############################################################

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


prompt_mapping = {
    '1': 'c_incorrect',
    '2': 'c_correct',
    '3': 'ambiguous',
    '4': 'bb_incorrect'
}




def prompt_img(bbox_indexes, epo_df, default_df, default_df_csvpath):

    print(f'Number of data to process = {len(bbox_indexes)}\n')

    counter = 0

    for index, row in epo_df.iterrows():

        if index not in bbox_indexes:
            continue

        if (default_df.at[index, 'c_incorrect'] != 'Not Detected' and default_df.at[index, 'c_correct'] != 'Not Detected' and default_df.at[index, 'ambiguous'] != 'Not Detected' and default_df.at[index, 'bb_incorrect'] != 'Not Detected'):
            print(f'***** [WARNING] {index} is already processed *****')
            continue

        try:

            img = np.array(Image.open(row['imgpath']))

            img_with_bbox = draw_bboxes(img, 
                                        [row['bbox_gt']],
                                        [row['label_gt']],
                                        [row['label_gt_des']],
                                        [row['label_pred']],
                                        [row['label_pred_des']], 
                                        name2red,
                                        name2white, 
                                        isshow=False)
            
            img_shape = (img.shape[0], img.shape[1])
            img_cropped = crop_bboxes(img_with_bbox, img_shape, [row['bbox_gt']], [row['label_gt']], pad=40)
            
            print(f"(#{index}){row['imgpath']}\ngt={row['label_gt_des']}, pred={row['label_pred_des']}")
            counter += 1

            val = {
                '1': False,
                '2': False,
                '3': False,
                '4': False
            }

            def onpress(event):
                
                sys.stdout.flush()

                if event.key in prompt_mapping.keys():

                    if val[event.key] is False:
                        val[event.key] = True
                    else:
                        val[event.key] = False

                    print(f'event.key={event.key}    {prompt_mapping[event.key]}={val[event.key]}')
                    
                if event.key == 'enter':
                    plt.cla()
                    plt.close()

            # with plt.ioff():
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 14))

            # fig.canvas.mpl_connect('button_press_event', onclick)
            fig.canvas.mpl_connect('key_press_event', onpress)

            ax1.axis('off')
            ax2.axis('off')
            ax1.imshow(img_with_bbox)
            ax2.imshow(img_cropped[0])


            ax1.set_title(f"No.{counter}(#{index}) gt({row['label_gt']}): {row['label_gt_des']}\npred({row['label_pred']}): {row['label_pred_des']}\nc_incorrect=1, c_correct=2 ambiguous=3, bb_incorrect=4", y=-0.2, fontsize=14)

            plt.tight_layout()
        
            disconnect_zoom = zoom_factory(ax1)
            pan_handler = panhandler(fig)

            plt.show()

            if len(val) == 0:
                print('Skip')
                plt.cla()
                plt.close()
                print('--------------------------')
                continue


            submit(default_df, default_df_csvpath, index, val)
            print('--------------------------')

        except Exception as e:
            print(f'Error: {e}')
            print(traceback.format_exc())
            break
    

def submit(df, df_csvpath, row_index, val):

    # Remove duplicated columns
    issue_columns = [prompt_mapping[k] for k, v in val.items() if v == True]

    print(f'On submitted: {issue_columns}')

    df.loc[row_index, 'c_incorrect'] = 'FALSE'
    df.loc[row_index, 'c_correct'] = 'FALSE'
    df.loc[row_index, 'ambiguous'] = 'FALSE'
    df.loc[row_index, 'bb_incorrect'] = 'FALSE'
    
    # for col in issue_columns:
    df.loc[row_index, issue_columns] = 'TRUE'

    print(f'Saved index={row_index}')


    saved_columns = ['c_incorrect', 'c_correct', 'ambiguous', 'bb_incorrect']

    df[saved_columns].to_csv(f'{df_csvpath}', index=True)

    plt.cla()
    plt.close()



def main():

    epochs = list(range(1, 13))
    # nsplits = list(range(0, 10))
    # epochs = [8,9,10,11,12]
    # nsplits = [0, 1]
    nsplits = ['01']


    # is_issue_types = ['gt incorrect', 'gt correct but ambiguous', 'gt correct', 'ambiguous', 'not detected']
    
    # methods = ['PBNR_NMargin','PBNR_SConf','PBNR_CWE','CL_NMargin','CL_SConf','CL_CWE','ArgMax_NMargin','ArgMax_SConf','ArgMax_CWE','LRank']

    # colnames = ['PBNR_NMargin','PBNR_SConf', 'PBNR_CWE']
    colnames = ['CL_NM','CL_SC','SC', 'NM', 'LRank']

    if 10 in id2class.keys():
        del id2class[10]

    classnames = list(id2class.values())

    for epoch in epochs:
        for nsplit in nsplits:

            csv_dirpath = f'./postprocessing/code/DataFrame/BDD100K'
            # csv_epo_dirpath = f'{csv_dirpath}/swin-t_val_split01'
            csv_epo_dirpath = f'{csv_dirpath}/val_split01'

            print(f'Start processing epoch={epoch} nsplit={nsplit}')

            csv_filename = f'epo{epoch}.csv'

            csv_default_filepath = f'./postprocessing/code/DataFrame/BDD100K/FasterRCNN/default_val_s{nsplit}.csv'

            epo_df = pd.read_csv(f'{csv_epo_dirpath}/{csv_filename}', index_col=0)
            epo_df['prob'] = epo_df['prob'].apply(lambda x: np.asarray(x.split(';'), dtype=float))
            epo_df['bbox_gt'] = epo_df['bbox_gt'].apply(lambda x: np.asarray(x.split(';'), dtype=float))

            ###########################################################
            #               Add New Temporary Columns:                #
            #      c_incorrect, ambiguous, c_correct, bb_incorrect    #
            #          Options: 'False', 'True', 'Not Detected'       #
            ###########################################################

            if not os.path.exists(f'{csv_default_filepath}'):
                print('No default csv file found. Creating one...')
                # empty_df = pd.DataFrame(columns=['c_incorrect', 'c_correct', 'ambiguous', 'bb_incorrect'])
                # empty_df = empty_df.assign(c_incorrect='Not Detected')
                # empty_df = empty_df.assign(c_correct='Not Detected')
                # empty_df = empty_df.assign(ambiguous='Not Detected')
                # empty_df = empty_df.assign(bb_incorrect='Not Detected')
                # default_df = empty_df
            else:
                print('Default csv file found. Loading...')
                default_df = pd.read_csv(f'{csv_default_filepath}', index_col=0, header=0)
            
            topn = 30
            union_idxs = compare_idxes(epo_df, colnames, topn, classnames, by_class=False)
            union_idxs_byclass = compare_idxes(epo_df, colnames, topn, classnames, by_class=True)

            print(f'Number of total union = {len(union_idxs)}, union_idxs_byclass={len(union_idxs_byclass)}')

            # union_idxs = union_idxs.union(union_idxs_byclass)

            ###############################################################
            #                 Number of Detected data                     #
            ###############################################################

            not_detected_conds = (default_df['c_incorrect'] == 'Not Detected') & (default_df['c_correct'] == 'Not Detected') & (default_df['ambiguous'] == 'Not Detected') & (default_df['bb_incorrect'] == 'Not Detected')
            df_detected = default_df.loc[~not_detected_conds]

            # df_detected = default_df.loc[~((default_df['c_incorrect'] == 'Not Detected') & 
            #                     (default_df['c_correct'] == 'Not Detected')&
            #                     (default_df['ambiguous'] == 'Not Detected')&
            #                     (default_df['bb_incorrect'] == 'Not Detected'))]
            
            print(f'Original default file, number of already detected data = {len(df_detected)} out of {len(default_df)} ({100*len(df_detected)/len(default_df):.3f}%)')
            indexes_detected = set(df_detected.index.tolist())
            tobe_checked_idxs = set(union_idxs) - indexes_detected
            print(f'This epo{epoch} add new to be checked data = #{len(tobe_checked_idxs)} out of {len(default_df)} ({100*len(tobe_checked_idxs)/len(default_df):.3f}%)')

            ###########################################################
            #                Loss Ranking  (deprecated)               #
            ###########################################################
            # # print('Compare CL label issues without Loss Ranking')

            # # take top N number of items (N is from CL method union)
            # top_loss_indexes = epo_df.loc[:, 'loss'].sort_values(ascending=False)[:len(union_idxs)].index.tolist()

            # tobe_checked_idxs = set(top_loss_indexes) | set(union_idxs) - set(indexes_detected)
            # print(f'This epo{epoch} add new to be checked data = #{len(tobe_checked_idxs)} out of {len(default_df)} {100*len(tobe_checked_idxs)/len(default_df):.3f}%')

            ###############################################################
            # print(f'tobe_checked_idxs={tobe_checked_idxs}')
            tobe_checked_idxs = [31370, 24330, 21518, 19345, 14866, 22423, 28055, 36376, 4763, 18464, 26401, 14880, 12326, 28209, 16306, 17457, 32824, 36936, 6985, 10700, 21198, 21327, 19153, 23003, 9055, 15461, 19813, 4711, 10345, 20972, 7918, 16626, 14586, 5115, 7292, 2046, 3839]
            prompt_img(tobe_checked_idxs, epo_df, default_df, csv_default_filepath)

            print('-----------------------------')


############################################################
#                     Run CL methods                       #
############################################################
ranked_by_options = ['self_confidence', 'normalized_margin', 'confidence_weighted_entropy']
filter_by_options = ['prune_by_class', 'prune_by_noise_rate', 'both', 'confident_learning', 'predicted_neq_given', 'low_normalized_margin', 'low_self_confidence']

# CSV columns, index is unique index for each cropped images
# ,prob,max_prob,label_gt,label_gt_des,label_pred,label_pred_des,loss,imgpath,imgname,PBNR_NMargin,PBNR_SConf,PBNR_CWE,CL_NMargin,CL_SConf,CL_CWE,ArgMax_NMargin,ArgMax_SConf,ArgMax_CWE,LRank

# https://docs.cleanlab.ai/v2.0.0/cleanlab/filter.html?highlight=predicted_neq_given#cleanlab.filter.find_label_issues

shorten = {
    'normalized_margin': 'NMargin',
    'self_confidence': 'SConf',
    'confidence_weighted_entropy': 'CWE',
    'prune_by_class': 'PBC',
    'prune_by_noise_rate': 'PBNR',
    'both': 'Both',
    'confident_learning': 'CL',
    'predicted_neq_given':  'ArgMax',
    'Loss_Ranking': 'LRank'
}

###############################################################################
#                              prune_by_noise_rate                            #
###############################################################################
# filter_by_option = 'prune_by_noise_rate'
# classnames = list(id2class.values())


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


def add_CL_ranking(filter_by, bbox_indexes, bbox_gt_labels, bbox_probs):

    tmp_df = pd.DataFrame()
    # dic = {}

    for ranked_by in ranked_by_options:
        
        frac_noise = 1.0
        label_issues = get_label_issues(bbox_gt_labels, bbox_probs, frac_noise, filter_by, ranked_by)

        print(f'ranked_by={ranked_by}, label_issues(#={len(label_issues)})={label_issues[:5]}')


        key = f'{shorten[filter_by]}_{shorten[ranked_by]}_new'
        print(f'Add {key} to bbox_dic')

        label_issues_set = set(label_issues)
        not_detected_indexes = set(bbox_indexes) - label_issues_set

        
        # allcolnames = ['prob','max_prob','bbox_gt','label_gt','label_gt_des','label_pred','label_pred_des','loss','imgpath', key]

        # combined_nsplit_df = pd.DataFrame(columns=allcolnames)

        rankings = []
        indexes = []

        for ranking, bbox_idx in zip(range(len(label_issues)), label_issues):
            rankings.append(ranking)
            indexes.append(bbox_idx)

        for ranking, bbox_idx in zip([999999]*len(not_detected_indexes), not_detected_indexes):
            rankings.append(ranking)
            indexes.append(bbox_idx)

        

        # dic[key] = rankings
        tmp_series = pd.Series(data=rankings, index=indexes)

        tmp_df.insert(len(tmp_df.columns), key, tmp_series)

        print('----')

        # for ranking, idx in enumerate(label_issues):
            # dic[idx][key] = ranking


        # for idx in bbox_dic.keys():
        #     if idx in not_detected_indexes:
        #         dic[idx][key] = 999999
        #     assert dic[idx][key] is not None

    print(tmp_df.head(4))
        
    return tmp_df.sort_index(axis=0)


def add_loss_ranking(df):
    loss_ranking_index = df['loss'].sort_values(ascending=False, inplace=False).index
    loss_ranking_mapper = {idx: ranking for ranking, idx in enumerate(loss_ranking_index)}

    df['LRank'] = df.index.to_series().map(loss_ranking_mapper)

    max_loss_idx = df['loss'].argmax()
    assert df['LRank'][max_loss_idx] == 0, 'max loss index is not 0'

    return df


def combine_nsplits_default():


    csv_dirpath = f'./postprocessing/code/DataFrame/BDD100K/FasterRCNN'
    default_0 = f'{csv_dirpath}/default_val_s0.csv'
    default_1 = f'{csv_dirpath}/default_val_s1.csv'
    target_default_csvpath = f'{csv_dirpath}/default_val_s01.csv'

    default_df0 = pd.read_csv(f'{default_0}', index_col=0, header=0)
    default_df1 = pd.read_csv(f'{default_1}', index_col=0, header=0)

    combined_default_df = pd.concat([default_df0, default_df1], axis=0, join='outer', ignore_index=True)

    combined_default_df.to_csv(f'{target_default_csvpath}', index=True, header=True)



def combine_nsplits():

    epochs = list(range(1, 13))
    # nsplits = list(range(0, 10))
    # epochs = [12]
    nsplits = [0, 1]

    


    # is_issue_types = ['gt incorrect', 'gt correct but ambiguous', 'gt correct', 'ambiguous', 'not detected']
    
    methods = ['PBNR_NMargin','PBNR_SConf','PBNR_CWE','CL_NMargin','CL_SConf','CL_CWE','ArgMax_NMargin','ArgMax_SConf','ArgMax_CWE','LRank']

    # issue_types = ['c_incorrect','c_correct','ambiguous','bb_incorrect']

    # colnames = ['PBNR_NMargin','PBNR_SConf', 'PBNR_CWE']

    combined_dic = {epo: None for epo in epochs}

    for epoch in epochs:
        print(f'=================== epoch={epoch} ====================')
        allcolnames = ['prob','max_prob','bbox_gt','label_gt','label_gt_des','label_pred','label_pred_des','loss','imgpath','imgname'] + methods

        combined_nsplit_df  = pd.DataFrame(columns=allcolnames)

        for nsplit in nsplits:

            csv_dirpath = f'./postprocessing/code/DataFrame/BDD100K/FasterRCNN'
            csv_epo_dirpath = f'{csv_dirpath}/e{epoch}'

            print(f'Start processing epoch={epoch} nsplit={nsplit}')

            csv_filename = f'e{epoch}_val_s{nsplit}.csv'

            # csv_default_filepath = f'{csv_dirpath}/default_val_s{nsplit}.csv'

            epo_df = pd.read_csv(f'{csv_epo_dirpath}/{csv_filename}', index_col=0)
            epo_df['prob'] = epo_df['prob'].apply(lambda x: np.asarray(x.split(';'), dtype=float))
            epo_df['bbox_gt'] = epo_df['bbox_gt'].apply(lambda x: np.asarray(x.split(';'), dtype=float))

            print('Found Loading default csv file...')
            # default_df = pd.read_csv(f'{csv_default_filepath}', index_col=0)

            # assert len(epo_df) == len(default_df), 'Length of epo_df and default_df are not equal'

            concated_df = epo_df.copy()
            # concated_df = pd.concat([epo_df, default_df], axis=1)

            combined_nsplit_df = pd.concat([combined_nsplit_df, concated_df], axis=0, join='outer', ignore_index=True)
            combined_dic[epoch] = combined_nsplit_df

        combined_df = combined_dic[epoch]
        bbox_indexes = combined_df.index.values.astype('int64')
        bbox_gt_labels = combined_df['label_gt'].values.astype('int64')
        bbox_probs = np.stack(combined_df['prob'].values).astype('float64')

        pbnr_df = add_CL_ranking('prune_by_noise_rate', bbox_indexes, bbox_gt_labels, bbox_probs)
        cl_df = add_CL_ranking('confident_learning', bbox_indexes, bbox_gt_labels, bbox_probs)
        png_df = add_CL_ranking('predicted_neq_given', bbox_indexes, bbox_gt_labels, bbox_probs)


        combined_df = pd.concat([combined_df, pbnr_df, cl_df, png_df], axis=1, join='outer', ignore_index=False)
    
        
        # ranked_by_options = ['prune_by_noise_rate', 'confident_learning', 'predicted_neq_given']

        for method in methods:

            if method == 'LRank':
                # LRank = 
                # # np.argsort(combined_df['loss'].values, axis=0)[::-1]
                # combined_df['LRank_new'] = LRank
                continue

            old_idxs = set(combined_df[method].values.tolist())
            new_idxs = set(combined_df[f'{method}_new'].values.tolist())
            
            print(f'method={method}')
            print(f'old_idxs={len(old_idxs)}')
            print(f'new_idxs={len(new_idxs)}')
            diff_idxs = new_idxs - old_idxs
            print(f'diff_idxs={len(diff_idxs)}')
            print('-----')

        saved_df = combined_df.copy()
        saved_df['prob'] = saved_df['prob'].apply(lambda x: ';'.join([f'{i:.16}' for i in x]))
        saved_df['bbox_gt'] = saved_df['bbox_gt'].apply(lambda x: ';'.join([f'{i:.16}' for i in x]))

        rename_mapper = {
            'PBNR_NMargin_new': 'PBNR_NMargin',
            'PBNR_SConf_new': 'PBNR_SConf',
            'PBNR_CWE_new': 'PBNR_CWE',
            'CL_NMargin_new': 'CL_NMargin',
            'CL_SConf_new': 'CL_SConf',
            'CL_CWE_new': 'CL_CWE',
            'ArgMax_NMargin_new': 'ArgMax_NMargin',
            'ArgMax_SConf_new': 'ArgMax_SConf',
            'ArgMax_CWE_new': 'ArgMax_CWE'
        }
        saved_df = saved_df.drop(['PBNR_NMargin','PBNR_SConf','PBNR_CWE','CL_NMargin','CL_SConf','CL_CWE','ArgMax_NMargin','ArgMax_SConf','ArgMax_CWE','LRank'], axis=1)

        saved_df = saved_df.rename(columns=rename_mapper)
        saved_df = add_loss_ranking(saved_df)
        saved_df.to_csv(f'{csv_dirpath}/e{epoch}/e{epoch}_val_s01.csv', index=True)
    print('=====================')

    # for colname in colnames:
    #     prompt_img(bbox_indexes, df, default_df, default_df_csvpath)


def prompt_classifier_img(bbox_indexes, df, default_df, default_df_csvpath):

    print(f'Number of data to process = {len(bbox_indexes)}\n')

    counter = 0

    for index, row in df.iterrows():

        if index not in bbox_indexes:
            continue

        if (default_df.at[index, 'c_incorrect'] != 'Not Detected' and default_df.at[index, 'c_correct'] != 'Not Detected' and default_df.at[index, 'ambiguous'] != 'Not Detected'):
            print(f'***** [WARNING] {index} is already processed *****')
            continue

        try:

            img = np.array(Image.open(row['imgpath']))

            counter += 1

            val = {
                '1': False,
                '2': False,
                '3': False
            }

            def onpress(event):
                
                sys.stdout.flush()

                if event.key in prompt_mapping.keys():

                    if val[event.key] is False:
                        val[event.key] = True
                    else:
                        val[event.key] = False

                    print(f'event.key={event.key}    {prompt_mapping[event.key]}={val[event.key]}')
                    
                if event.key == 'enter':
                    plt.cla()
                    plt.close()

            # with plt.ioff():
            fig, (ax1) = plt.subplots(1, 1, figsize=(6, 6))

            # fig.canvas.mpl_connect('button_press_event', onclick)
            fig.canvas.mpl_connect('key_press_event', onpress)

            ax1.axis('off')
            ax1.imshow(img)


            ax1.set_title(f"No.{counter}(#{index}) gt({row['label_gt']}): {row['label_gt_des']}\npred({row['label_pred']}): {row['label_pred_des']}\nc_incorrect=1, c_correct=2 ambiguous=3", y=-0.25, fontsize=14)

            plt.tight_layout()
        
            disconnect_zoom = zoom_factory(ax1)
            pan_handler = panhandler(fig)

            plt.show()

            if len(val) == 0:
                print('Skip')
                plt.cla()
                plt.close()
                print('--------------------------')
                continue


            classifier_submit(default_df, default_df_csvpath, index, val)
            print('--------------------------')

        except Exception as e:
            print(f'Error: {e}')
            print(traceback.format_exc())
            break
    

def classifier_submit(df, df_csvpath, row_index, val):

    # Remove duplicated columns
    issue_columns = [prompt_mapping[k] for k, v in val.items() if v == True]

    print(f'On submitted: {issue_columns}')

    df.loc[row_index, 'c_incorrect'] = 'FALSE'
    df.loc[row_index, 'c_correct'] = 'FALSE'
    df.loc[row_index, 'ambiguous'] = 'FALSE'
    
    # for col in issue_columns:
    df.loc[row_index, issue_columns] = 'TRUE'

    print(f'Saved index={row_index}')


    saved_columns = ['c_incorrect', 'c_correct', 'ambiguous']

    df[saved_columns].to_csv(f'{df_csvpath}', index=True)

    plt.cla()
    plt.close()


def run_classifier_checker():

    epochs = list(range(10, 61, 10))
    # epochs = [10]

    # m_name = 'CNN2'
    m_name = 'TRANS'
    csv_source_dirpath = f'./postprocessing/code/DataFrame/swedish'
    csv_default_filepath = f'{csv_source_dirpath}/swe_global_issue.csv'

    # is_issue_types = ['c_correct', 'ambiguous', 'c_incorrect']
    
    methods = ['PBNR_NMargin','PBNR_SConf','PBNR_CWE','CL_NMargin','CL_SConf','CL_CWE','ArgMax_NMargin','ArgMax_SConf','ArgMax_CWE','LRank']

    # target_methods = ['PBNR_NMargin','PBNR_SConf', 'PBNR_CWE']
    target_methods = ['CL_NMargin','CL_SConf','CL_CWE', 'LRank']

    classnames = list(id2class.values())

    for epoch in epochs:

        csv_epo_filepath = f'{csv_source_dirpath}/{m_name}/swe_{m_name}_epo{epoch}.csv'
        print(f'Start processing epoch={epoch}, {csv_epo_filepath}')

        default_df = pd.read_csv(f'{csv_default_filepath}', index_col=0, header=0)
        epo_df = pd.read_csv(f'{csv_epo_filepath}', index_col=0, header=0)
        epo_df['prob'] = epo_df['prob'].apply(lambda x: np.asarray(x.split(';'), dtype=float))

        assert len(epo_df) == len(default_df), 'Length of epo_df and default_df are not equal'

        topn = 200
        union_idxs = compare_idxes(epo_df, target_methods, topn, classnames, by_class=False)
        union_idxs_byclass = compare_idxes(epo_df, target_methods, topn, classnames, by_class=True)

        print(f'Number of total union = {len(union_idxs)}, union_idxs_byclass={len(union_idxs_byclass)}')

        union_idxs = union_idxs.union(union_idxs_byclass)

        ###############################################################
        #                 Number of Detected data                     #
        ###############################################################

        df_detected = default_df.loc[~((default_df['c_incorrect'] == 'Not Detected') & 
                            (default_df['c_correct'] == 'Not Detected')&
                            (default_df['ambiguous'] == 'Not Detected'))]
        
        print(f'Original default file, number of not detected data = {len(df_detected)} out of {len(default_df)} {100*len(df_detected)/len(default_df):.3f}%')
        indexes_detected = set(df_detected.index.tolist())
        tobe_checked_idxs = set(union_idxs) - indexes_detected
        print(f'This epo{epoch} add new to be checked data = #{len(tobe_checked_idxs)} out of {len(default_df)} {100*len(tobe_checked_idxs)/len(default_df):.3f}%')

        ###########################################################
        #                       Loss Ranking                      #
        ###########################################################
        # print('Compare CL label issues without Loss Ranking')

        # # take top N number of items (N is from CL method union)
        # top_loss_indexes = epo_df.loc[:, 'loss'].sort_values(ascending=False)[:len(union_idxs)].index.tolist()

        # tobe_checked_idxs = (set(top_loss_indexes) | set(union_idxs)) - set(indexes_detected)
        # print(f'This epo{epoch} add new to be checked data = #{len(tobe_checked_idxs)} out of {len(default_df)} {100*len(tobe_checked_idxs)/len(default_df):.3f}%')

        # ###############################################################
        # print(f'tobe_checked_idxs={tobe_checked_idxs}')
        # if len(tobe_checked_idxs) > 0:
        #     prompt_classifier_img(tobe_checked_idxs, epo_df, default_df, csv_default_filepath)

        print('-----------------------------')


def random_classifier_checker():

    # m_name = 'CNN2'
    m_name = 'TRANS'

    csv_source_dirpath = f'./postprocessing/code/DataFrame/swedish'
    csv_default_filepath = f'{csv_source_dirpath}/swe_global_issue.csv'
    default_df = pd.read_csv(f'{csv_default_filepath}', index_col=0, header=0)

    all_indexes = default_df.index.tolist()
    pick_ratio = 0.1
    pick_n = int(len(all_indexes) * pick_ratio)

    random.seed(42)
    picked_indexes = random.choices(all_indexes, k=pick_n)
    print(f'picked_indexes:')
    print(picked_indexes)

    # epoch = 10
    # csv_epo_filepath = f'{csv_source_dirpath}/{m_name}/swe_{m_name}_epo{epoch}.csv'
    # print(f'Start processing epoch={epoch}, {csv_epo_filepath}')

    # default_df = pd.read_csv(f'{csv_default_filepath}', index_col=0, header=0)
    # epo_df = pd.read_csv(f'{csv_epo_filepath}', index_col=0, header=0)
    # epo_df['prob'] = epo_df['prob'].apply(lambda x: np.asarray(x.split(';'), dtype=float))

    # prompt_classifier_img(picked_indexes, epo_df, default_df, csv_default_filepath)

    e_condition = default_df['c_incorrect'] == 'TRUE'
    ea_condition = (default_df['ambiguous'] == 'TRUE') & (default_df['c_correct'] == 'FALSE')

    e_df = default_df.loc[e_condition, :]
    ea_df = default_df.loc[ea_condition, :]

    print(f'len(e_df)={len(e_df)}')
    print(f'len(ea_df)={len(ea_df)}')

    print(f'Precision of e_df = {len(e_df.loc[e_df["c_correct"] == "FALSE", :]) / len(e_df):.3f}')

    print(f'How many E level label errors of picked items: {100.0*len(e_df)/len(picked_indexes):.3f}%')
    print(f'How many EA level label errors of picked item: {100.0*len(ea_df)/len(picked_indexes):.3f}%')

    print(f'How many E level label errors of all: {100.0*len(e_df)/len(default_df):.3f}%')
    print(f'How many EA level label errors of all: {100.0*len(ea_df)/len(default_df):.3f}%')



    ###############################################################
    #                 Number of Detected data                     #
    ###############################################################

    # df_detected = default_df.loc[~((default_df['c_incorrect'] == 'Not Detected') & 
    #                     (default_df['c_correct'] == 'Not Detected')&
    #                     (default_df['ambiguous'] == 'Not Detected'))]
    
    # print(f'Original default file, number of detected data = {len(df_detected)} out of {len(default_df)} {100*len(df_detected)/len(default_df):.3f}%')



def  check_bdd100k_gdino():

    # is_issue_types = ['gt incorrect', 'gt correct but ambiguous', 'gt correct', 'ambiguous', 'not detected']
    
    colnames = ['CL_NM','CL_SC','SC', 'NM']

    if 10 in id2class.keys():
        del id2class[10]

    classnames = list(id2class.values())

    dirname = 'bdd100k_gdino_swint'

    csv_dirpath = f'./postprocessing/code/DataFrame/bdd100k_gdino_swint'
    # csv_epo_dirpath = f'{csv_dirpath}/swin-t_val_split01'
    # csv_epo_dirpath = f'{csv_dirpath}/split01.csv'

    # print(f'Start processing epoch={epoch} nsplit={nsplit}')

    csv_filename = f'split01.csv'

    csv_default_filepath = f'./postprocessing/code/DataFrame/BDD100K/FasterRCNN/default_val_s01.csv'

    epo_df = pd.read_csv(f'{csv_dirpath}/{csv_filename}', index_col=0)
    epo_df['prob'] = epo_df['prob'].apply(lambda x: np.asarray(x.split(';'), dtype=float))
    epo_df['bbox_gt'] = epo_df['bbox_gt'].apply(lambda x: np.asarray(x.split(';'), dtype=float))

    ###########################################################
    #               Add New Temporary Columns:                #
    #      c_incorrect, ambiguous, c_correct, bb_incorrect    #
    #          Options: 'False', 'True', 'Not Detected'       #
    ###########################################################

    if not os.path.exists(f'{csv_default_filepath}'):
        print('No default csv file found. Creating one...')
        # empty_df = pd.DataFrame(columns=['c_incorrect', 'c_correct', 'ambiguous', 'bb_incorrect'])
        # empty_df = empty_df.assign(c_incorrect='Not Detected')
        # empty_df = empty_df.assign(c_correct='Not Detected')
        # empty_df = empty_df.assign(ambiguous='Not Detected')
        # empty_df = empty_df.assign(bb_incorrect='Not Detected')
        # default_df = empty_df
    else:
        print('Default csv file found. Loading...')
        default_df = pd.read_csv(f'{csv_default_filepath}', index_col=0, header=0)
    
    topn = 200
    union_idxs = compare_idxes(epo_df, colnames, topn, classnames, by_class=False)
    union_idxs_byclass = compare_idxes(epo_df, colnames, topn, classnames, by_class=True)

    print(f'Number of total union = {len(union_idxs)}, union_idxs_byclass={len(union_idxs_byclass)}')

    # union_idxs = union_idxs.union(union_idxs_byclass)

    ###############################################################
    #                 Number of Detected data                     #
    ###############################################################

    not_detected_conds = (default_df['c_incorrect'] == 'Not Detected') & (default_df['c_correct'] == 'Not Detected') & (default_df['ambiguous'] == 'Not Detected') & (default_df['bb_incorrect'] == 'Not Detected')
    df_detected = default_df.loc[~not_detected_conds]

    # df_detected = default_df.loc[~((default_df['c_incorrect'] == 'Not Detected') & 
    #                     (default_df['c_correct'] == 'Not Detected')&
    #                     (default_df['ambiguous'] == 'Not Detected')&
    #                     (default_df['bb_incorrect'] == 'Not Detected'))]
    
    print(f'Original default file, number of already detected data = {len(df_detected)} out of {len(default_df)} ({100*len(df_detected)/len(default_df):.3f}%)')
    indexes_detected = set(df_detected.index.tolist())
    tobe_checked_idxs = set(union_idxs) - indexes_detected
    print(f'Add new to be checked data = #{len(tobe_checked_idxs)} out of {len(default_df)} ({100*len(tobe_checked_idxs)/len(default_df):.3f}%)')

    ###########################################################
    #                Loss Ranking  (deprecated)               #
    ###########################################################
    # # print('Compare CL label issues without Loss Ranking')

    # # take top N number of items (N is from CL method union)
    # top_loss_indexes = epo_df.loc[:, 'loss'].sort_values(ascending=False)[:len(union_idxs)].index.tolist()

    # tobe_checked_idxs = set(top_loss_indexes) | set(union_idxs) - set(indexes_detected)
    # print(f'This epo{epoch} add new to be checked data = #{len(tobe_checked_idxs)} out of {len(default_df)} {100*len(tobe_checked_idxs)/len(default_df):.3f}%')

    ###############################################################
    # print(f'tobe_checked_idxs={tobe_checked_idxs}')
    # tobe_checked_idxs = [31370, 24330, 21518, 19345, 14866, 22423, 28055, 36376, 4763, 18464, 26401, 14880, 12326, 28209, 16306, 17457, 32824, 36936, 6985, 10700, 21198, 21327, 19153, 23003, 9055, 15461, 19813, 4711, 10345, 20972, 7918, 16626, 14586, 5115, 7292, 2046, 3839]
    prompt_img(tobe_checked_idxs, epo_df, default_df, csv_default_filepath)

    print('-----------------------------')


if __name__ == "__main__":
    #################################
    #        Object Detection       #
    #################################
    # main()
    check_bdd100k_gdino()
    # combine_nsplits_default()
    # combine_nsplits()
    #################################
    #     Image Classificatoin      #
    #################################
    # run_classifier_checker()
    # random_classifier_checker()
    