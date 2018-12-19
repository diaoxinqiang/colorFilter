# coding:utf8

import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import uuid
import sys
sys.setrecursionlimit(1000000)


def get_random_position(init, addition, max, random_min=20, random_max=200):
    random = int(init + np.random.randint(random_min, random_max))
    if (random + addition > max):
        return get_random_position(init, addition, max, random_min, random_max)
    else:
        return random


# 图像处理，将logo图标叠加到一张图片的右上角，要求有颜色的区域为不透明
def img_deal(background_path, foreground_labels, images, save_path):
    img_target = cv2.imread(background_path, cv2.IMREAD_COLOR)
    target_height, target_width, channel1 = img_target.shape
    # file_name = os.path.splitext(os.path.basename(background_path))[0]
    file_name = str(uuid.uuid4())
    file_name = '{}_{}.jpg'.format(file_name, 'train')
    distance_x = int(target_width / 3)
    distance_y = int(target_height / 5)
    positions_x = np.zeros(3)
    positions_y = np.zeros(5)
    for index, position_x in enumerate(positions_x):
        positions_x[index] = distance_x * index
    for index, position_y in enumerate(positions_y):
        positions_y[index] = distance_y * index

    image = {}
    image['file_attributes'] = {}
    image['filename'] = file_name
    regions = []
    image['regions'] = regions
    for index, foreground_label in enumerate(foreground_labels):
        label_name = '{}.jpg'.format(foreground_label['file_name'])
        foreground_path = os.path.join('./label/all/', label_name)
        img_foreground = cv2.imread(foreground_path, cv2.IMREAD_COLOR)

        # 2，对logo做清洗，白色区域是255，其他区域置为黑色0
        img_foreground_gray = cv2.cvtColor(img_foreground, cv2.COLOR_BGR2GRAY)
        ret, img_foreground_mask = cv2.threshold(img_foreground_gray, 20, 255,
                                                 cv2.THRESH_BINARY)  # 二值化函数
        img_logo_mask1 = cv2.bitwise_not(img_foreground_mask)
        img_foreground[img_foreground_mask == 0] = 255

        # 3，提取目标图片的ROI （region of interest）感兴趣区域
        foreground_height, foreground_width, channel = img_foreground.shape
        from_y = get_random_position(
            int(positions_y[int(index / 3)]), foreground_height, target_height, 10, 300)
        from_x = get_random_position(
            int(positions_x[index % 3]), foreground_width, target_width, 10, 300)

        to_y = from_y + foreground_height
        to_x = from_x + foreground_width
        img_roi = img_target[from_y:to_y,
                  from_x:to_x].copy()
        cv2.imshow("img_roi", img_roi)

        # 4，ROI和Logo图像融合
        img_res0 = cv2.bitwise_and(img_roi, img_roi, mask=img_logo_mask1)
        img_res1 = cv2.bitwise_and(img_foreground, img_foreground, mask=img_foreground_mask)
        img_res2 = cv2.add(img_res0, img_res1)
        # img_res2 = img_res0 + img_res1
        img_target[from_y:to_y,
        from_x:to_x] = img_res2[:, :]

        region = {}

        region_attributes = {}
        if foreground_label['type'] == 0:
            region_attributes['name'] = 'right'
        elif foreground_label['type'] == 1:
            region_attributes['name'] = 'wrong'
        # region_attributes['type'] = foreground_label['type']
        region['region_attributes'] = region_attributes

        shape_attributes = {}
        all_points_x = []
        all_points_y = []
        for index, position in enumerate(foreground_label['regions']):
            # if index % 2 == 0 or index % 3 == 0 or index % 5 == 0:
            #     continue
            all_points_x.append(int(position[0] + from_x))
            all_points_y.append(int(position[1] + from_y))

        shape_attributes['all_points_x'] = all_points_x
        shape_attributes['all_points_y'] = all_points_y
        shape_attributes['name'] = 'polygon'
        region['shape_attributes'] = shape_attributes

        regions.append(region)

    image_path = os.path.join(save_path, file_name)
    cv2.imwrite(image_path, img_target)

    file_size = os.path.getsize(image_path)
    image['size'] = int(file_size)
    images[file_name + str(image['size'])] = image
    return image


# 显示图片，调用opencv展示
# cv2.imshow("img_res0", img_res0)
# cv2.imshow("img_res1", img_res1)
# cv2.imshow("img_res2", img_res2)
# cv2.imshow("img_target", img_target)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 显示图片，调用matplotlib展示
# plt.figure()
# titles = ["img_logo", "img_logo_gray", "img_logo_mask", "img_logo_mask1", "img_roi", "img_res0",
#           "img_res1",
#           "img_res2"]
# imgs = [img_logo, img_logo_gray, img_logo_mask, img_logo_mask1, img_roi, img_res0, img_res1,
#         img_res2]
# for x in range(len(imgs)):
#     plt.subplot(241 + x), plt.imshow(img_convert(imgs[x]), cmap='gray'), plt.title(
#         titles[x])  # , plt.axis('off')
# plt.show()

# 显示图片，调用matplotlib展示
# plt.figure()
# plt.subplot(332), plt.imshow(img_convert(img_res2), cmap='gray'), plt.title("img_res2")
# plt.subplot(323), plt.imshow(img_convert(img_res0), cmap='gray'), plt.title("img_res0")
# plt.subplot(324), plt.imshow(img_convert(img_res1), cmap='gray'), plt.title("img_res1")
# plt.subplot(3, 4, 9), plt.imshow(img_convert(img_roi), cmap='gray'), plt.title("img_roi")
# plt.subplot(3, 4, 10), plt.imshow(img_convert(img_logo_mask), cmap='gray'), plt.title(
#     "img_logo_mask")
# plt.subplot(3, 4, 11), plt.imshow(img_convert(img_logo), cmap='gray'), plt.title("img_logo")
# plt.subplot(3, 4, 12), plt.imshow(img_convert(img_logo_mask1), cmap='gray'), plt.title(
#     "img_logo_mask1")
# plt.show()

# cv2与matplotlib的图像转换，cv2是bgr格式，matplotlib是rgb格式


def img_convert(cv2_img):
    # 灰度图片直接返回
    if len(cv2_img.shape) == 2:
        return cv2_img
    # 3通道的BGR图片
    elif len(cv2_img.shape) == 3 and cv2_img.shape[2] == 3:
        b, g, r = cv2.split(cv2_img)
        return cv2.merge((r, g, b))
    # 4通道的BGR图片
    elif len(cv2_img.shape) == 3 and cv2_img.shape[2] == 4:
        b, g, r, a = cv2.split(cv2_img)
        return cv2.merge((r, g, b, a))
    # 未知图片格式
    else:
        return cv2_img


# 主函数
if __name__ == "__main__":
    img_deal()
