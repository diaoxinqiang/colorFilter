import cv2
import numpy as np
import sys
import os
import glob
import copy
import colorFilter as colorFilter
from PIL import Image, ImageEnhance
import json


def generate_mask_points(image_path, type=0):
    file_name = os.path.splitext(os.path.basename(image_path))[0]

    # 读取图片
    image = cv2.imread(image_path, flags=cv2.IMREAD_COLOR)
    _, threshold_image = cv2.threshold(image, 20, 255, cv2.THRESH_BINARY)

    mask_gray = cv2.cvtColor(threshold_image, cv2.COLOR_BGR2GRAY)
    ret, mask_binary = cv2.threshold(mask_gray, 0, 255, cv2.THRESH_BINARY)
    mask_image, mask_contours, hierarchy = cv2.findContours(mask_binary, cv2.RETR_TREE,
                                                            cv2.CHAIN_APPROX_SIMPLE)
    label = {}
    label['file_name'] = file_name
    regions = []
    label['width'] = image.shape[0]
    label['height'] = image.shape[1]
    label['type'] = type
    for index, contour in enumerate(mask_contours):
        regions += contour[0:contour.shape[0], 0].tolist()
    label['regions'] = regions

    cv2.drawContours(image, mask_contours, -1, (0, 255, 0), 1)  # 标记处编号为0的轮廓

    # colorFilter.show_image('mask_image', image)

    return label


def save_json(path, data):
    with open(path, 'w') as f:
        f.write(json.dumps(data))


if __name__ == '__main__':
    img_dir_path = './label/right/'
    img_list = glob.glob(os.path.join(img_dir_path, '*.jp*'))
    print('处理图片数量:{}'.format(len(img_list)))
    labels = []
    label_type = 0
    for image_path in img_list:
        label = generate_mask_points(image_path, label_type)
        labels.append(label)
    path = './label/right/labels.json'
    save_json(path, labels)
    print(len(labels))
