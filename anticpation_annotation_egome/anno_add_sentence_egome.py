# coding: utf-8

import os
import json

def filter_and_concatenate_descriptions(start_time, end_time, annotations):
    """
    根据开始时间是否落在标注的时间区间内，筛选并拼接描述
    
    参数:
        start_time (float): 开始时间（只检查这个时间是否落在某个标注的区间内）
        end_time (float): 结束时间（不再用于区间重叠检查）
        annotations (list): 标注列表，每个标注是一个字典
        
    返回:
        str: 符合条件的描述拼接后的字符串
    """
    matched_descriptions = []
    
    for annotation in annotations:
        step_start, step_end = annotation['Step timestamp']
        
        # 检查开始时间是否落在当前标注的时间区间内
        if step_start <= ((start_time + end_time) / 2.0) <= step_end:
            matched_descriptions.append(annotation['Step discription'])  # 注意：原代码拼写错误，应为 'description'
        
    if len(matched_descriptions) == 0:
        matched_descriptions.append('.')
    
    # 将匹配的描述用空格连接起来
    return ' '.join(matched_descriptions)




with open('/data1/zhaofeng/TTA/anti-TTA/action_anticipation_planning_benchmark_v16_coop/anticpation_annotation_egome/EgoMe_Annotation_total.json', 'r') as f:
    egome_anno_data = json.load(f)['annotations']

# print(egome_anno_data.keys())

ori_dirs_list = ['noun', 'verb']


for ori_dir in ori_dirs_list:
    for ori_anno_file in os.listdir(ori_dir):
        if 'test' not in ori_anno_file:
            continue

        with open(os.path.join(ori_dir, ori_anno_file),'r', encoding='utf-8') as f1:
            ori_anno_datas = f1.readlines()


        ## save
        data_add_text = []
        target_dir = ori_dir + '_addtext'
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        
        for ori_anno_data in ori_anno_datas:
            ori_anno_data = ori_anno_data.rstrip()

            ori_file_name = ori_anno_data.split('|')[0] + '.mp4'
            ori_start_time = float(ori_anno_data.split('|')[1]) 
            ori_end_time = float(ori_anno_data.split('|')[2])

            # 计算观察区间的起止时间
            ori_observe_end_time = ori_start_time - 1
            ori_observe_start_time = ori_observe_end_time - 2

            # egome data
            egome_anno_data_item = egome_anno_data[ori_file_name]['Fine-level']
            description_item = filter_and_concatenate_descriptions(ori_observe_start_time, ori_observe_end_time, egome_anno_data_item)

            after_anno_data = ori_anno_data + '|' + description_item + '\n'

            data_add_text.append(after_anno_data)
    

        # 打开文件并写入（'w' 表示写入模式，会覆盖原有内容）
        with open(os.path.join(target_dir,ori_anno_file), "w") as f3:
            for line in data_add_text:
                f3.write(line) 
            