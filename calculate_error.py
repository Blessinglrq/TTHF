from __future__ import division
import numpy as np
import scipy.signal as signal
import joblib
import json
import os
from sklearn.metrics import roc_curve, auc


def normalize_video(video_scores):
    video_scores_max = video_scores.max()
    video_scores_min = video_scores.min()
    if video_scores_max - video_scores_min > 0:
        video_scores = (video_scores - video_scores_min) / (video_scores_max - video_scores_min)
    return video_scores


def compute_tad_scores(scores, label, args, sub_test=True, dataset='dota'):
    dota_meta = joblib.load(
        open(os.path.join('/data/lrq/DoTA/mnt/workspace/datasets/DoTA_dataset', "ground_truth_demo/gt_metadata.json"), "rb"))
    meta_data = json.load(
        open(os.path.join('/data/lrq/DoTA/mnt/workspace/datasets/DoTA_dataset', "metadata_val.json"), "rb"))
    data_meta = joblib.load(
    open(os.path.join('/data/lrq/DADA-2000', "ground_truth_demo/gt_metadata.json"), "rb"))
    # for dada dataset
    ego_involved_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11',
                         '12', '13', '14', '15', '16', '17', '18', '51', '52']

    gt_concat = np.concatenate(list(label.values()), axis=0)
    new_gt = np.array([])
    new_frame_scores = np.array([])
    gt_video_frame_nums = []
    start_idx = 0
    if args.other_method in ['TDAFF_BASE']:
        video_idx = 0
    if sub_test:
        if dataset == 'dota':
            gt_subclass = {'ST': np.array([]), 'AH': np.array([]), 'LA': np.array([]), 'OC': np.array([]), 'TC': np.array([]),
                           'VP': np.array([]), 'OO': np.array([]), 'VO': np.array([]), 'UK': np.array([]), 'ST*': np.array([]),
                           'AH*': np.array([]), 'LA*': np.array([]),
                           'OC*': np.array([]), 'TC*': np.array([]), 'VP*': np.array([]), 'VO*': np.array([]),
                           'OO*': np.array([]), 'UK*': np.array([])}
            score_subclass = {'ST': np.array([]), 'AH': np.array([]), 'LA': np.array([]), 'OC': np.array([]), 'TC': np.array([]),
                           'OO': np.array([]), 'VO': np.array([]), 'VP': np.array([]), 'UK': np.array([]), 'ST*': np.array([]), 'AH*': np.array([]), 'LA*': np.array([]),
                           'OC*': np.array([]), 'TC*': np.array([]), 'VP*': np.array([]), 'VO*': np.array([]),
                           'OO*': np.array([]), 'UK*': np.array([])}
            gt_sub_video_frame_nums = {'ST': [], 'AH': [], 'LA': [], 'OC': [], 'TC': [],
                           'OO': [], 'VO': [], 'VP': [], 'UK': [], 'ST*': [], 'AH*': [], 'LA*': [],
                           'OC*': [], 'TC*': [], 'VP*': [], 'VO*': [],
                           'OO*': [], 'UK*': []}
        elif dataset == 'dada':
            gt_subclass = {'ego_involved': np.array([]), 'non_ego': np.array([])}
            score_subclass = {'ego_involved': np.array([]), 'non_ego': np.array([])}
            gt_sub_video_frame_nums = {'ego_involved': [], 'non_ego': []}

    for cur_video_id in range(len(list(label.values()))):
        cur_video_len = len(list(label.values())[cur_video_id])
        gt_video_frame_nums.append(cur_video_len)
        gt_each_video = gt_concat[start_idx:start_idx + cur_video_len]
        if args.other_method in ['TDAFF_BASE']:
            scores_each_video = np.concatenate(([0], scores[video_idx:video_idx + cur_video_len - 1]))
        scores_each_video = signal.medfilt(scores_each_video, kernel_size=27)  # 95?
        scores_each_video = normalize_video(scores_each_video)

        new_gt = np.concatenate((new_gt, gt_each_video), axis=0)
        new_frame_scores = np.concatenate((new_frame_scores, scores_each_video), axis=0)
        if sub_test:
            if dataset == 'dota':
                if meta_data[dota_meta['testing_video'][cur_video_id]]['anomaly_class'] == 'other: moving_ahead_or_waiting':
                    gt_subclass['AH*'] = np.concatenate((gt_subclass['AH*'], gt_each_video), axis=0)
                    score_subclass['AH*'] = np.concatenate((score_subclass['AH*'], scores_each_video), axis=0)
                    gt_sub_video_frame_nums['AH*'].append(cur_video_len)
                elif meta_data[dota_meta['testing_video'][cur_video_id]]['anomaly_class'] == 'other: turning':
                    gt_subclass['TC*'] = np.concatenate((gt_subclass['TC*'], gt_each_video), axis=0)
                    score_subclass['TC*'] = np.concatenate((score_subclass['TC*'], scores_each_video), axis=0)
                    gt_sub_video_frame_nums['TC*'].append(cur_video_len)
                elif meta_data[dota_meta['testing_video'][cur_video_id]]['anomaly_class'] == 'other: lateral':
                    gt_subclass['LA*'] = np.concatenate((gt_subclass['LA*'], gt_each_video), axis=0)
                    score_subclass['LA*'] = np.concatenate((score_subclass['LA*'], scores_each_video), axis=0)
                    gt_sub_video_frame_nums['LA*'].append(cur_video_len)
                elif meta_data[dota_meta['testing_video'][cur_video_id]]['anomaly_class'] == 'other: leave_to_left' or \
                        meta_data[dota_meta['testing_video'][cur_video_id]]['anomaly_class'] == 'other: leave_to_right':
                    gt_subclass['OO*'] = np.concatenate((gt_subclass['OO*'], gt_each_video), axis=0)
                    score_subclass['OO*'] = np.concatenate((score_subclass['OO*'], scores_each_video), axis=0)
                    gt_sub_video_frame_nums['OO*'].append(cur_video_len)
                elif meta_data[dota_meta['testing_video'][cur_video_id]]['anomaly_class'] == 'other: obstacle':
                    gt_subclass['VO*'] = np.concatenate((gt_subclass['VO*'], gt_each_video), axis=0)
                    score_subclass['VO*'] = np.concatenate((score_subclass['VO*'], scores_each_video), axis=0)
                    gt_sub_video_frame_nums['VO*'].append(cur_video_len)
                elif meta_data[dota_meta['testing_video'][cur_video_id]]['anomaly_class'] == 'other: start_stop_or_stationary':
                    gt_subclass['ST*'] = np.concatenate((gt_subclass['ST*'], gt_each_video), axis=0)
                    score_subclass['ST*'] = np.concatenate((score_subclass['ST*'], scores_each_video), axis=0)
                    gt_sub_video_frame_nums['ST*'].append(cur_video_len)
                elif meta_data[dota_meta['testing_video'][cur_video_id]]['anomaly_class'] == 'other: oncoming':
                    gt_subclass['OC*'] = np.concatenate((gt_subclass['OC*'], gt_each_video), axis=0)
                    score_subclass['OC*'] = np.concatenate((score_subclass['OC*'], scores_each_video), axis=0)
                    gt_sub_video_frame_nums['OC*'].append(cur_video_len)
                elif meta_data[dota_meta['testing_video'][cur_video_id]]['anomaly_class'] == 'other: pedestrian':
                    gt_subclass['VP*'] = np.concatenate((gt_subclass['VP*'], gt_each_video), axis=0)
                    score_subclass['VP*'] = np.concatenate((score_subclass['VP*'], scores_each_video), axis=0)
                    gt_sub_video_frame_nums['VP*'].append(cur_video_len)
                elif meta_data[dota_meta['testing_video'][cur_video_id]]['anomaly_class'] == 'other: unknown':
                    gt_subclass['UK*'] = np.concatenate((gt_subclass['UK*'], gt_each_video), axis=0)
                    score_subclass['UK*'] = np.concatenate((score_subclass['UK*'], scores_each_video), axis=0)
                    gt_sub_video_frame_nums['UK*'].append(cur_video_len)
                elif meta_data[dota_meta['testing_video'][cur_video_id]]['anomaly_class'] == 'ego: moving_ahead_or_waiting':
                    gt_subclass['AH'] = np.concatenate((gt_subclass['AH'], gt_each_video), axis=0)
                    score_subclass['AH'] = np.concatenate((score_subclass['AH'], scores_each_video), axis=0)
                    gt_sub_video_frame_nums['AH'].append(cur_video_len)
                elif meta_data[dota_meta['testing_video'][cur_video_id]]['anomaly_class'] == 'ego: turning':
                    gt_subclass['TC'] = np.concatenate((gt_subclass['TC'], gt_each_video), axis=0)
                    score_subclass['TC'] = np.concatenate((score_subclass['TC'], scores_each_video), axis=0)
                    gt_sub_video_frame_nums['TC'].append(cur_video_len)
                elif meta_data[dota_meta['testing_video'][cur_video_id]]['anomaly_class'] == 'ego: lateral':
                    gt_subclass['LA'] = np.concatenate((gt_subclass['LA'], gt_each_video), axis=0)
                    score_subclass['LA'] = np.concatenate((score_subclass['LA'], scores_each_video), axis=0)
                    gt_sub_video_frame_nums['LA'].append(cur_video_len)
                elif meta_data[dota_meta['testing_video'][cur_video_id]]['anomaly_class'] == 'ego: leave_to_left' or \
                        meta_data[dota_meta['testing_video'][cur_video_id]]['anomaly_class'] == 'ego: leave_to_right':
                    gt_subclass['OO'] = np.concatenate((gt_subclass['OO'], gt_each_video), axis=0)
                    score_subclass['OO'] = np.concatenate((score_subclass['OO'], scores_each_video), axis=0)
                    gt_sub_video_frame_nums['OO'].append(cur_video_len)
                elif meta_data[dota_meta['testing_video'][cur_video_id]]['anomaly_class'] == 'ego: obstacle':
                    gt_subclass['VO'] = np.concatenate((gt_subclass['VO'], gt_each_video), axis=0)
                    score_subclass['VO'] = np.concatenate((score_subclass['VO'], scores_each_video), axis=0)
                    gt_sub_video_frame_nums['VO'].append(cur_video_len)
                elif meta_data[dota_meta['testing_video'][cur_video_id]]['anomaly_class'] == 'ego: start_stop_or_stationary':
                    gt_subclass['ST'] = np.concatenate((gt_subclass['ST'], gt_each_video), axis=0)
                    score_subclass['ST'] = np.concatenate((score_subclass['ST'], scores_each_video), axis=0)
                    gt_sub_video_frame_nums['ST'].append(cur_video_len)
                elif meta_data[dota_meta['testing_video'][cur_video_id]]['anomaly_class'] == 'ego: oncoming':
                    gt_subclass['OC'] = np.concatenate((gt_subclass['OC'], gt_each_video), axis=0)
                    score_subclass['OC'] = np.concatenate((score_subclass['OC'], scores_each_video), axis=0)
                    gt_sub_video_frame_nums['OC'].append(cur_video_len)
                elif meta_data[dota_meta['testing_video'][cur_video_id]]['anomaly_class'] == 'ego: pedestrian':
                    gt_subclass['VP'] = np.concatenate((gt_subclass['VP'], gt_each_video), axis=0)
                    score_subclass['VP'] = np.concatenate((score_subclass['VP'], scores_each_video), axis=0)
                    gt_sub_video_frame_nums['VP'].append(cur_video_len)
                elif meta_data[dota_meta['testing_video'][cur_video_id]]['anomaly_class'] == 'ego: unknown':
                    gt_subclass['UK'] = np.concatenate((gt_subclass['UK'], gt_each_video), axis=0)
                    score_subclass['UK'] = np.concatenate((score_subclass['UK'], scores_each_video), axis=0)
                    gt_sub_video_frame_nums['UK'].append(cur_video_len)
            elif dataset == 'dada':
                if data_meta['testing_video'][cur_video_id].split('_')[0] in ego_involved_list:
                    gt_subclass['ego_involved'] = np.concatenate((gt_subclass['ego_involved'], gt_each_video), axis=0)
                    score_subclass['ego_involved'] = np.concatenate((score_subclass['ego_involved'], scores_each_video), axis=0)
                    gt_sub_video_frame_nums['ego_involved'].append(cur_video_len)
                else:
                    gt_subclass['non_ego'] = np.concatenate((gt_subclass['non_ego'], gt_each_video), axis=0)
                    score_subclass['non_ego'] = np.concatenate((score_subclass['non_ego'], scores_each_video),
                                                                    axis=0)
                    gt_sub_video_frame_nums['non_ego'].append(cur_video_len)
        start_idx += cur_video_len
        if args.other_method in ['TDAFF_BASE']:
            video_idx += cur_video_len - 1

    gt_concat = new_gt
    frame_scores = new_frame_scores

    curves_save_path = os.path.join(args.model_dir, args.exp_name, 'anomaly_curves')
    auc = save_evaluation_curves(frame_scores, gt_concat, curves_save_path,
                                 np.array(gt_video_frame_nums))
    if sub_test:
        for key in gt_subclass.keys():
            sub_auc = save_evaluation_curves(score_subclass[key], gt_subclass[key], curves_save_path,
                                     np.array(gt_sub_video_frame_nums[key]))
            print('AUC: ', key, sub_auc)
    return auc


def save_evaluation_curves(scores, labels, curves_save_path, video_frame_nums):
    """
    Draw anomaly score curves for each video and the overall ROC figure.
    """
    if not os.path.exists(curves_save_path):
        os.mkdir(curves_save_path)

    scores = scores.flatten()
    labels = labels.flatten()

    scores_each_video = {}
    labels_each_video = {}

    start_idx = 0
    for video_id in range(len(video_frame_nums)):
        scores_each_video[video_id] = scores[start_idx:start_idx + video_frame_nums[video_id]]
        labels_each_video[video_id] = labels[start_idx:start_idx + video_frame_nums[video_id]]

        start_idx += video_frame_nums[video_id]

    truth = []
    preds = []
    for i in range(len(scores_each_video)):
        truth.append(labels_each_video[i])
        preds.append(scores_each_video[i])

    truth = np.concatenate(truth, axis=0)
    preds = np.concatenate(preds, axis=0)
    fpr, tpr, roc_thresholds = roc_curve(truth, preds, pos_label=1)
    auroc = auc(fpr, tpr)
    return auroc
