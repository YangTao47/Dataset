''''*!
    * \date 2021/12/4
    *
    * \author Yang, Tao
    * Contact: 627871875@qq.com
    *
    *
    * \note
*'''
import argparse
import glob
import json
import os
import os.path as osp
import shutil
import numpy as np


def find_files_with_suffix(target_dir, target_suffix="json"):
    """ 查找以 target_suffix 为后缀的文件，并返加 """
    find_res = []
    target_suffix_dot = "." + target_suffix
    walk_generator = os.walk(target_dir)
    for root_path, dirs, files in walk_generator:
        if len(files) < 1:
            continue
        for file in files:
            # shutil.move(os.path.join(root_path, file), os.path.join('new_path', file))
            file_name, suffix_name = os.path.splitext(file)
            if suffix_name == target_suffix_dot:
                find_res.append(os.path.join(root_path, file))
    return find_res


def images_labelme(data):
    image = {}
    image['id'] = str(data["id"])
    image['height'] = data['height']
    image['width'] = data['width']
    '''if '\\' in data['imagePath']:
        image['file_name'] = data['imagePath'].split('\\')[-1]
    else:
        image['file_name'] = data['imagePath'].split('/')[-1]'''
    label = get_points(data)
    image['label'] = label
    return image


def get_points(data):
    label = []
    num = len(data['labels'])
    for i in range(num):
        tmp = {}
        if data['labels'][i]['shape_type'] == 'rectangle':
            tmp['label'] = data['labels'][i]['label']
        tmp['points'] = data['labels'][i]['points']
        tmp['shape_type'] = data['labels'][i]['shape_type']
        label.append(tmp)
    return label

import random
all_json = find_files_with_suffix("B:\\meter-data\\meter_seg\\2021-12-27-new")
all = os.listdir("B:\\meter-data\\meter_seg\\2021-12-27-new")
random.shuffle(all_json)
all_label = []
with open("B:\\meter-data\\meter_seg\\2021-12-27-new.json", "w") as fp:
    for file in all_json:
        with open(file) as f:
            data = json.load(f)
            image = images_labelme(data)
            # fp.write(json.dumps(image, ensure_ascii=False, indent=1))
            all_label.append(image)
    json.dump(all_label, fp, indent=2)


