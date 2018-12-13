import cv2
import numpy as np
import sys
import os
import glob
import copy


def leave_color(img, lower, upper):
    """
    通过剥离图片为：筛选出只含目标颜色的图片
    :param img:图片
    :param lower:最小颜色值域 HSV值
    :param upper: 最大颜色值域 HSV值
    :return: 过滤后的图片
    """
    try:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        for i in range(lower.shape[0]):
            mask = + cv2.inRange(hsv, lower[i], upper[i])
        _img = cv2.bitwise_and(img, img, mask=mask)

        return _img
    except  Exception as e:
        print("分离图片颜色失败 %s", e)


def findContours(img):
    # img = cv2.imread('1024.jpg')
    # imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # ret,thresh = cv2.threshold(imgray,127,255,0)
    # image ,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # #绘制独立轮廓，如第四个轮廓
    # # contour = cv2.drawContour(img,contours,-1,(0,255,0),3)
    # #但是大多数时候，下面方法更有用
    # contour = cv2.drawContours(img,contours,3,(0,255,0),3)
    img = img.copy()
    # _, threshold_image = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    contour_image, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE,
                                                          cv2.CHAIN_APPROX_SIMPLE)

    print('轮廓坐标点数量:{}'.format(np.size(contours)))  # 得到该图中总的轮廓数量
    print('轮廓前五个坐标点:\n{}'.format(contours[0][0:5]))  # 打印出第一个轮廓的所有点的坐标， 更改此处的0，为0--（总轮廓数-1），可打印出相应轮廓所有点的坐标
    print('轮廓之间的关系:{}'.format(hierarchy))  # 打印出相应轮廓之间的关系
    img = cv2.drawContours(img, contours, -1, (0, 255, 0), 1)  # 标记处编号为0的轮廓

    contour_path = './redpen/'
    contour_path = "{0}{1}_contour.png".format(contour_path, 'contour')
    cv2.imwrite(contour_path, img)
    while (1):
        # cv2.imshow('img',img)
        # cv2.imshow('imgray',imgray)
        # cv2.imshow('image',image)
        cv2.imshow('contour', img)
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()


def generate_label_image(path):
    file_name = os.path.splitext(os.path.basename(path))[0]
    # 红色的hsv值：0-10，146-180
    lower_red = np.array([[0, 43, 46], [156, 43, 46]])
    upper_red = np.array([[10, 255, 255], [180, 255, 255]])
    # 读取图片
    image = cv2.imread(path, flags=cv2.IMREAD_COLOR)
    img_BGRA = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
    # 图片如果太大，缩小四倍
    # image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    # cv2.imshow("original", image)
    # 过滤其他颜色，只留下红色
    filter_image = leave_color(image, lower_red, upper_red)
    # 变成灰度图片
    tmp = cv2.cvtColor(filter_image, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(filter_image)
    rgba = [b, g, r, alpha]
    img_with_alpha = cv2.merge(rgba)
    findContours(img_with_alpha)
    # 二值化 得到比较干净的红笔痕迹
    # _, threshold_image = cv2.threshold(filter_image, 0, 255, cv2.THRESH_BINARY)

    label_path = './redpen/'
    label_path = "{0}{1}_label.png".format(label_path, file_name)
    cv2.imwrite(label_path, img_with_alpha)

    # cv2.imshow('target', img_with_alpha)
    # # 按q 退出
    # while True:
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         print
    #         "I'm done"
    #         break


if __name__ == '__main__':
    img_dir_path = './'
    img_list = glob.glob(os.path.join(img_dir_path, '*.jp*'))
    print('处理图片数量:{}'.format(len(img_list)))
    for image_path in img_list:
        generate_label_image(image_path)
