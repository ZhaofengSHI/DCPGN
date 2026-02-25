import os

def txt_to_list(filename, delimiter=None):
    """
    将文本文件转换为Python列表
    :param filename: 文件名
    :param delimiter: 分隔符(可选)
    :return: 内容列表
    """
    with open(filename, 'r', encoding='utf-8') as file:
        if delimiter:
            content = file.read()
            return [item.strip() for item in content.split(delimiter)]
        else:
            return [line.strip() for line in file.readlines()]




ori_file = '/data1/zhaofeng/TTA/anti-TTA/action_anticipation_planning_benchmark_v16_coop_text/anticpation_annotation_eel_filtered/verb/list_ego_test_ego_exo-feature.txt'

new_file = '/data1/zhaofeng/TTA/anti-TTA/action_anticipation_planning_benchmark_v16_coop_text/anticpation_annotation_eel_filtered/verb_addtext/list_ego_test_ego_exo-feature.txt'

ori_list = txt_to_list(ori_file)
new_list = txt_to_list(new_file)


for i in range(len(new_list)):

    new_item_front = '|'.join(new_list[i].split('|')[:-1])

    if new_item_front != ori_list[i]:
        print(i+1)



