import random

import cv2
import numpy as np
import sys
import os
import glob
import copy
import colorFilter as colorFilter
from PIL import Image, ImageEnhance
import json
import merge as merge
# -*- coding:utf-8 -*-
import json
import contourJsonGenerater as contourJsonGenerater

right_json_path = './label/right/labels.json'
wrong_json_path = './label/wrong/labels.json'


def generate_train_image(path, images, save_path):
    right_labels = read_json(right_json_path)
    wrong_labels = read_json(wrong_json_path)
    for right in right_labels:
        right['type'] = 0
    for wrong in wrong_labels:
        wrong['type'] = 1
    contourJsonGenerater.save_json(right_json_path, right_labels)
    contourJsonGenerater.save_json(wrong_json_path, wrong_labels)
    foreground_labels = random.sample(right_labels, 5)
    foreground_labels.extend(random.sample(wrong_labels, 10))
    # right_labels[np.random.random_integers(0, len(right_labels))] + \
    random.shuffle(foreground_labels)
    merge.img_deal(path, foreground_labels, images, save_path)


def read_json(file_path):
    with open(file_path) as f:
        json_data = json.load(f)  # js是转换后的字典
        return json_data


if __name__ == '__main__':
    data_path = 'train'
    background_dir_path = os.path.join('./background/', data_path)
    background_img_paths = glob.glob(os.path.join(background_dir_path, '*.jp*'))
    images = {}
    save_path = os.path.join('./dataset/', data_path)
    for background_img_path in background_img_paths:
        generate_train_image(background_img_path, images, save_path)

    images_json_path = os.path.join(save_path, 'via_region_data.json')
    contourJsonGenerater.save_json(images_json_path, images)
