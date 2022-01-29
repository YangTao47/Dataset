''''*!
    * \date 2021/12/4
    *
    * \author Yang, Tao
    * Contact: 627871875@qq.com
    *
    *
    * \note
*'''
import numpy as np
import cv2
import os
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables.kps import Keypoint, KeypointsOnImage
import imgaug as ia
import time
import json
import numpy

nums = 1
idx = 0
#5430 10860 16820 22780 29080
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (numpy.int_, numpy.intc, numpy.intp, numpy.int8,
                            numpy.int16, numpy.int32, numpy.int64, numpy.uint8,
                            numpy.uint16, numpy.uint32, numpy.uint64)):
            return int(obj)
        elif isinstance(obj, (numpy.float_, numpy.float16, numpy.float32, numpy.float64)):
            return float(obj)
        elif isinstance(obj, (numpy.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def find_files_with_suffix(target_dir, target_suffix="jpg"):
    """ 查找以 target_suffix 为后缀的文件，并返加 """
    find_jpg = []
    find_json = []
    id = []
    walk_generator = os.walk(target_dir)
    for root_path, dirs, files in walk_generator:
        if len(files) < 1:
            continue
        for file in files:
            # shutil.move(os.path.join(root_path, file), os.path.join('new_path', file))
            file_name, suffix_name = os.path.splitext(file)
            if suffix_name == ".jpg":
                find_jpg.append(os.path.join(root_path, file))
                id.append(file_name)
            if suffix_name == ".json":
                find_json.append(os.path.join(root_path, file))

    return find_jpg, find_json, id


def get_kp(json):
    kp = []
    for i in range(len(json['shapes'])):
        if json['shapes'][i]['shape_type'] == 'point':
            kp.append(json['shapes'][i]['points'][0])
    return kp


def get_bodbx(json):
    bodbx = []
    for i in range(6):
        if json['shapes'][i]['shape_type'] == 'rectangle':
            bodbx.append(json['shapes'][i]['points'])
    return bodbx


# 取出当前秒
a = time.gmtime().tm_sec
# 设置随机数种子
ia.seed(time.gmtime().tm_sec)


def encode_points(data):
    label = []
    num = len(data['shapes'])
    for i in range(num):
        tmp = {}
        tmp['label'] = data['shapes'][i]['label']
        tmp['points'] = data['shapes'][i]['points']
        tmp['shape_type'] = data['shapes'][i]['shape_type']
        label.append(tmp)
    return label


def get_imgaug(data, img_path, img_id):
    global nums, idx
    # 读取数据
    kp = get_kp(data)
    bodbx = get_bodbx(data)
    # 读取图片
    img = cv2.imread(img_path + '\\' + str(img_id) + '.jpg')
    # 变换通道
    img = img[:, :, ::-1]
    kps = KeypointsOnImage([
        Keypoint(x=kp[0][0], y=kp[0][1]),
        Keypoint(x=kp[1][0], y=kp[1][1]),
        Keypoint(x=kp[2][0], y=kp[2][1]),
        Keypoint(x=kp[3][0], y=kp[3][1]),
        Keypoint(x=kp[4][0], y=kp[4][1]),
        Keypoint(x=kp[5][0], y=kp[5][1])
    ], shape=img.shape)
    bbs = BoundingBoxesOnImage([
        # 目标在图片上的位置
        BoundingBox(x1=bodbx[0][0][0], y1=bodbx[0][0][1], x2=bodbx[0][1][0], y2=bodbx[0][1][1]),
        BoundingBox(x1=bodbx[1][0][0], y1=bodbx[1][0][1], x2=bodbx[1][1][0], y2=bodbx[1][1][1])
    ], shape=img.shape)
    # 数据增强
    seq = iaa.SomeOf(2, [
        iaa.LinearContrast((0.8, 1.2)),
        # iaa.Fliplr(0.5),
        #iaa.Flipud(0.5),
        iaa.ImpulseNoise(0.05),
        iaa.Salt(0.05),
        # iaa.CoarsePepper(0.02, size_percent=(0.05, 0.06)),
        # iaa.BlendAlpha((0.0, 1.0), foreground=iaa.Add(30), background=iaa.Multiply(0.1)),
        #iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
        iaa.GaussianBlur(sigma=(0, 1.5)),
        iaa.AverageBlur(k=((3, 3), (7, 7))),
        # iaa.Multiply((0.8, 1.2), per_channel=0.2),
        #iaa.ScaleX((0.8, 1.2)),
        #iaa.ScaleY((0.8, 1.2)),
        iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                   rotate=(-10, 10))], random_order=True)
    # 变换后的图片和box
    img_aug, bbs_aug, kps_aug = seq(image=img, bounding_boxes=bbs, keypoints=kps)
    cv2.imwrite("B:\\meter-data\\meter_seg\\2021-12-27-new\\" + str(idx + nums) + '.jpg', img_aug)
    # 绘制变换前box在图片上的位置
    '''img_before = bbs.draw_on_image(img, size=2)
    img_before = kps.draw_on_image(img, size=2)
    # 绘制图片变换后box在图片上的位置
    img_after = bbs_aug.draw_on_image(img_aug, size=2, color=[255, 0, 0])
    img_after = kps_aug.draw_on_image(img_after, size=2, color=[255, 0, 0])
    ia.show_grid([img_before, img_after], rows=1, cols=2)'''
    return img_aug, bbs_aug, kps_aug


def write_json_file(data, path, bbs_aug, kps_aug):
    json_data = {}
    label = []
    global nums, idx
    for i in range(len(kps_aug.items)):
        tmp = {}
        point = []
        point.append(kps_aug[i].x)
        point.append(kps_aug[i].y)
        tmp['points'] = point
        tmp['shape_type'] = "point"
        label.append(tmp)
    for i in range(len(bbs_aug.items)):
        tmp = {}
        points = []
        point1 = []
        point2 = []
        point1.append(bbs_aug[i].x1)
        point1.append(bbs_aug[i].y1)
        point2.append(bbs_aug[i].x2)
        point2.append(bbs_aug[i].y2)
        points.append(point1)
        points.append(point2)
        tmp['label'] = data['shapes'][i+4]['label']
        tmp['points'] = points
        tmp['shape_type'] = "rectangle"
        label.append(tmp)

    # json_data['version']='4.5.13'
    json_data['id'] = idx + nums
    json_data['height'] = data['imageHeight']
    json_data['width'] = data['imageWidth']
    # json_data['file_name'] = str(1000 + nums) + '.jpg'
    json_data['labels'] = label
    with open("B:\\meter-data\\meter_seg\\2021-12-27-new\\" + str(idx + nums) + '.json', 'w') as f:
        json.dump(json_data, f, indent=2, cls=NumpyEncoder)
    nums += 1


def dataEnforce(path, id, generateNum):
    # 读取数据
    with open(path + '\\' + str(id) + '.json') as fp:
        data = json.load(fp)
    for i in range(generateNum):
        _, bbs, kps = get_imgaug(data, path, id)
        write_json_file(data, path, bbs, kps)

def dataEnforcespecial(path, id, generateNum):
    # 读取数据
    with open(path + '\\' + str(id) + '.json') as fp:
        data = json.load(fp)
        for tmp in data['shapes']:
            if tmp['label'] == '25':
                for i in range(generateNum):
                    _, bbs, kps = get_imgaug(data, path, id)
                    write_json_file(data, path, bbs, kps)



all_jpg, all_json, id = find_files_with_suffix("B:\\meter-data\\meter_seg\\images\\newTrain")
for i in id:
    dataEnforce("B:\\meter-data\\meter_seg\\images\\newTrain", int(i), 20)



#1.6 150 10 46 25 182  66 217 54