import os
import json
import csv

import pandas as pd
from tqdm import tqdm
import math



# 读取 CSV 文件
eel_anno_data_csv = pd.read_csv('/data1/zhaofeng/TTA/anti-TTA/action_anticipation_planning_benchmark_v16_coop_text/anticpation_annotation_eel_filtered/fine_annotation_trainvaltest_en.csv')

# 预建索引（只需执行一次）
eel_anno_data_csv = eel_anno_data_csv.set_index('video_uid')
print(eel_anno_data_csv)

ori_dirs_list = ['noun', 'verb']

for ori_dir in ori_dirs_list:
    for ori_anno_file in os.listdir(ori_dir):
        if 'test' not in ori_anno_file:
            continue

        with open(os.path.join(ori_dir, ori_anno_file),'r', encoding='utf-8') as f1:
            ori_anno_datas = f1.readlines()

        print(ori_dir, ori_anno_file)
        ## save
        data_add_text = []
        target_dir = ori_dir + '_addtext'
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        for ori_anno_data in tqdm(ori_anno_datas):
            ori_anno_data = ori_anno_data.rstrip()

            ori_file_name = ori_anno_data.split('|')[0]
            ori_start_time = float(ori_anno_data.split('|')[1]) 
            ori_end_time = float(ori_anno_data.split('|')[2])

            # 计算观察区间的起止时间
            ori_observe_end_time = ori_start_time - 1
            ori_observe_start_time = ori_observe_end_time - 2

            # # 找对应文本xx
            matched_descriptions = []
            matched_rows = eel_anno_data_csv.loc[ori_file_name]

            ## 遍历原始标注中的数据
            for index, row in matched_rows.iterrows():

                # 如果落在某个区间内 保存相应文本
                if row['start_sec'] <= ((ori_observe_start_time + ori_observe_end_time) / 2.0) <= row['end_sec']:

                    if type(row['narration_en_no_hand_prompt']) == str:
                        matched_descriptions.append(row['narration_en_no_hand_prompt'])  # 注意：原代码拼写错误，应为 'description'
                    else:
                        print(ori_file_name)
                        print('abnormal')
                        print(row['narration_en_no_hand_prompt'])
                        matched_descriptions.append('.')

            if len(matched_descriptions) == 0:
                matched_descriptions.append('.')

            description_item = ' '.join(matched_descriptions)


            after_anno_data = ori_anno_data + '|' + description_item + '\n'
            data_add_text.append(after_anno_data)

        # 打开文件并写入（'w' 表示写入模式，会覆盖原有内容）
        with open(os.path.join(target_dir,ori_anno_file), "w") as f3:
            for line in data_add_text:
                f3.write(line) 



