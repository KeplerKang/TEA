#!/bin/bash

# fully_teacher_student
# 循环从 0.1 到 0.9，步长为 0.1
# for ratio in $(seq 0.1 0.1 1.0); do
#     echo "***************************  ratio: $ratio  ***************************"
#     save_path="tsseg/teastu/exps/0527_teastu/ratio_${ratio}"
#     python tea.py --config configs/PASTIS24/`TSViT_foldnew1.yaml \
#         --ratio ${ratio} \
#         --save_path ${save_path} \
#         --device 0,1,2,3 \
#         --describe 'teacher student single ratio train and test on all ratios, use foldnew1, 24-410 datasets, 64 batch size, 4 gpu, 100 epoch, '
# done

# save_path="tsseg/ours/exps/0712_tswin_teastu"
# python teacherstudent.py --config configs/PASTIS24-410/TSViT_fold3.yaml --ratio 0 --save_path ${save_path} --device 0,1,2,3,4,5,6,7

python train/baseline.py --config configs/PASTIS24-410/TSViT_fold1.yaml \
    --save_path tsseg/ours/exps/0724_tsvit --device 0,1,2,3


