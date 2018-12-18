# coding:utf8

import numpy as np
import cv2
from matplotlib import pyplot as plt


# 图像处理，将logo图标叠加到一张图片的右上角，要求有颜色的区域为不透明
def img_deal():
    img_logo = cv2.imread("fore_red.jpg", cv2.IMREAD_COLOR)

    # 2，对logo做清洗，白色区域是255，其他区域置为黑色0
    img_logo_gray = cv2.cvtColor(img_logo, cv2.COLOR_BGR2GRAY)
    ret, img_logo_mask = cv2.threshold(img_logo_gray, 20, 255, cv2.THRESH_BINARY)  # 二值化函数
    img_logo_mask1 = cv2.bitwise_not(img_logo_mask)
    img_logo[img_logo_mask == 0] = 255
    cv2.imshow("img_logo_gray", img_logo_gray)
    cv2.imshow("img_logo_mask", img_logo_mask)

    # 3，提取目标图片的ROI
    img_target = cv2.imread("bg_page.jpg", cv2.IMREAD_COLOR)
    rows, cols, channel = img_logo.shape
    rows1, cols1, channel1 = img_target.shape
    img_roi = img_target[:rows, cols1 - cols:cols1].copy()
    cv2.imshow("img_roi", img_roi)

    # 4，ROI和Logo图像融合
    img_res0 = cv2.bitwise_and(img_roi, img_roi, mask=img_logo_mask1)
    img_res1 = cv2.bitwise_and(img_logo, img_logo, mask=img_logo_mask)
    img_res2 = cv2.add(img_res0, img_res1)
    # img_res2 = img_res0 + img_res1
    img_target[:rows, cols1 - cols:cols1] = img_res2[:, :]
    cv2.imwrite("img_target.png", img_target)

    # 显示图片，调用opencv展示
    # cv2.imshow("img_res0", img_res0)
    # cv2.imshow("img_res1", img_res1)
    # cv2.imshow("img_res2", img_res2)
    # cv2.imshow("img_target", img_target)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 显示图片，调用matplotlib展示
    plt.figure()
    titles = ["img_logo", "img_logo_gray", "img_logo_mask", "img_logo_mask1", "img_roi", "img_res0",
              "img_res1",
              "img_res2"]
    imgs = [img_logo, img_logo_gray, img_logo_mask, img_logo_mask1, img_roi, img_res0, img_res1,
            img_res2]
    for x in range(len(imgs)):
        plt.subplot(241 + x), plt.imshow(img_convert(imgs[x]), cmap='gray'), plt.title(
            titles[x])  # , plt.axis('off')
    plt.show()

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
