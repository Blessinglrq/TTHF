#!/bin/bash
# extract gt
python3 ./extract_gt_label.py  # for DoTA dataset

python3 ./extract_gt_label_dada.py  # for DADA dataset

python3 ./extract_dota_one_hot_label.py  # extract fine-grained label

python3 ./extract_dota_one_hot_label_bi.py  # extract general label

python3 ./extract_dota_one_hot_label_val.py  # extract val label

python3 ./extract_dota_metadata.py  # extract meta_data
# FIXME extract for bbox
#python3 ./extract_bboxes_from_deep_sort.py \
#        --mode train \
#
#python3 ./extract_bboxes_from_deep_sort.py \
#        --mode test \

# FIXME extract for FOL and F2TAD
#python3 ./extract_samples_fol.py \
#        --mode train \
#        --L 11 \

#python3 ./extract_bbox_flow_fol.py \
#        --mode train \
#
#python3 ./extract_samples_f3tad.py \
#        --mode test \
#
#python3 ./extract_samples_f3tad.py \
#        --mode train \

## FIXME extract for fbf(ego_motion, fol)
#python3 ./extract_samples_fol.py \
#        --mode train \
#        --L 2 \
#
#python3 ./extract_samples_fol.py \
#        --mode test \
#        --L 2 \
#
#python3 ./extract_samples_f2tad_len_2.py \
#        --mode train \
#
#python3 ./extract_samples_f2tad_len_2.py \
#        --mode test \
#
### FIXME extract for end to end training
#python3 ./extract_samples_f3tad_fol.py \
#        --mode train \

#python3 ./extract_samples_f3tad_fol.py \
#        --mode test \

## FIXME extract for fbf(ego_motion, fol, obs=5)
#python3 ./extract_samples_fol.py \
#        --mode train \
#        --L 15 \
##
#python3 ./extract_samples_fol.py \
#        --mode test \
#        --L 15 \
#
#python3 ./extract_samples_f3tad_fol_15.py \
#        --mode train \
#
#python3 ./extract_samples_f3tad_fol_15.py \
#        --mode test \

#python3 ./extract_samples_f2tad_fol_15.py \
#        --mode train \
#
#python3 ./extract_samples_f2tad_fol_15.py \
#        --mode test \
#
#python3 ./extract_samples_f2tad_len_2.py \
#        --mode test \

#python3 ./extract_samples_f3tad_flow_fol_frame_15.py \
#        --mode train \

python3 ./extract_samples_f3tad_flow_fol_frame_15.py \
        --mode test \
