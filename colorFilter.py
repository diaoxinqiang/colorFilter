import cv2
import numpy as np
import sys
import os
import glob
import copy

from PIL import Image, ImageEnhance


def generate_label_image(path):
    file_name = os.path.splitext(os.path.basename(path))[0]

    # 读取图片
    image = cv2.imread(path, flags=cv2.IMREAD_COLOR)
    enhance_image = deal_with_image(image)
    # img_BGRA = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
    # 图片如果太大，缩小四倍
    # image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    # cv2.imshow("original", image)

    # 过滤其他颜色，只留下红色 该图用于寻找轮廓坐标
    # 模糊去除噪点
    blur_image = cv2.blur(image, (8, 8))
    # kernel = np.ones((5, 5), np.uint8)
    # image = cv2.GaussianBlur(image, (5, 3), 1)
    # bright_image = np.uint8(np.clip((1.5 * image + 10), 0, 255))
    for_mask_image = get_red_color(image, False)

    # 该图进行膨胀,为了得到框选对错号主体

    for_box_image = get_red_color(blur_image, True)

    # 根据box 裁剪出红笔痕迹图片
    boxing_image = generateRedBoxes(file_name, for_mask_image, for_box_image)

    # 二值化 得到比较干净的红笔痕迹
    # _, threshold_image = cv2.threshold(filter_image, 0, 255, cv2.THRESH_BINARY)

    label_path = "{0}{1}_label.png".format('./redpen/', file_name)
    image_data = np.vstack([image, for_mask_image, boxing_image])
    save_image(label_path, image_data)


def save_image(path, image):
    cv2.imwrite(path, image)


def show_image(name, image):
    while True:
        cv2.imshow(name, image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print
            "I'm done"
            break


def get_alpha_image(image):
    # 将所有 为0的值替换成255
    # image[image == 0] = 255

    # 变成灰度图片
    tmp = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #
    # # 将图像中灰度值为0的过滤成255
    _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
    # 分离rgb值
    b, g, r = cv2.split(image)
    # 合成含有透明度通道的图片,在有红笔痕迹的区域,透明度为0;其他区域透明度为255
    rgba = [b, g, r, alpha]
    img_with_alpha = cv2.merge(rgba)
    return img_with_alpha


def nothing(x):
    pass


def get_red_color(img, dilate=False):
    """
    通过剥离图片为：筛选出只含目标颜色的图片
    :param img:图片
    :param lower:最小颜色值域 HSV值
    :param upper: 最大颜色值域 HSV值
    :return: 过滤后的图片
    """
    try:
        # im_gray = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # 红色的hsv值：0-10，146-180
        # lower mask (0-10)
        lower_red = np.array([0, 43, 46])
        upper_red = np.array([10, 255, 255])
        mask0 = cv2.inRange(hsv, lower_red, upper_red)

        # upper mask (170-180)
        lower_red = np.array([146, 43, 46])
        upper_red = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)

        mask = mask0 + mask1
        if (dilate):
            kernel = np.ones((150, 150), np.uint8)
            mask = cv2.dilate(mask, kernel)

        output_img = cv2.bitwise_and(img, img, mask=mask)
        return output_img

    except  Exception as e:
        print("分离图片颜色失败 %s", e)


def generateRedBoxes(file_name, for_mask_image, for_box_image):
    mask_image = for_mask_image.copy()
    # img = cv2.imread('1024.jpg')
    # imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # ret,thresh = cv2.threshold(imgray,127,255,0)
    # image ,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # #绘制独立轮廓，如第四个轮廓
    # # contour = cv2.drawContour(img,contours,-1,(0,255,0),3)
    # #但是大多数时候，下面方法更有用
    # contour = cv2.drawContours(img,contours,3,(0,255,0),3)

    box_image = for_box_image.copy()
    # box_gray = cv2.cvtColor(box_image, cv2.COLOR_BGR2GRAY)
    box_gray = threshold_image(box_image)
    ret, box_binary = cv2.threshold(box_gray, 0, 255, cv2.THRESH_BINARY)
    box_image, box_contours, hierarchy = cv2.findContours(box_binary, cv2.RETR_TREE,
                                                          cv2.CHAIN_APPROX_SIMPLE)
    for index, contour in enumerate(box_contours):
        contour_path = './label/'
        contour_path = "{0}{1}_box_{2}.jpg".format(contour_path, 'right_02', index)
        # min_area_rectangle
        # min_rect = cv2.minAreaRect(contour)
        # img_crop = crop_minAreaRect(mask_image, min_rect)

        # cv2.imwrite(contour_path, img_crop)
        # min_rect = np.int0(cv2.boxPoints(min_rect))
        # cv2.drawContours(mask_image, [min_rect], 0, (255, 0, 0), 2)  # green

        x, y, w, h = cv2.boundingRect(contour)
        img_region = mask_image[y:y + h, x:x + w]
        save_image(contour_path, img_region)
        cv2.rectangle(for_box_image, (x, y), (x + w, y + h), (255, 0, 0), 8)

    # print('轮廓数量:{}'.format(np.size(box_contours)))  # 得到该图中总的轮廓数量
    # print('轮廓前五个坐标点:\n{}'.format(
    #     contours[0][0:5]))  # 打印出第一个轮廓的所有点的坐标， 更改此处的0，为0--（总轮廓数-1），可打印出相应轮廓所有点的坐标
    # print('轮廓之间的关系:{}'.format(hierarchy))  # 打印出相应轮廓之间的关系
    # contours = contours - np.ones(contours.contours)50
    # img = cv2.drawContours(for_mask_image, mask_contours, -1, (0, 255, 0), 4)  # 标记处编号为0的轮廓

    # contour_path = './redpen/'
    # contour_path = "{0}{1}_contour.png".format(contour_path, file_name)
    # cv2.imwrite(contour_path, img)

    # while (1):
    #     cv2.imshow('contour', img)
    #     if cv2.waitKey(1) == ord('q'):
    #         break
    return for_box_image
    cv2.destroyAllWindows()


def threshold_image(image):
    c = 185
    d = 15
    warped_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # # 双边滤波，去除iphone杂质，第一个参数不易多大，否则加大运算量
    thresh_gray = cv2.adaptiveThreshold(warped_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY,
                                        c, d)
    return thresh_gray


def deal_with_image(orign):
    thresh_gray = threshold_image(orign)
    # 创建空图像，并让每个颜色通道的二值化后的黑色部分用原图颜色替代（目的是让内容保持原来的颜色）
    res_img = np.zeros(orign.shape, np.uint8)
    res_img[:, :, 0] = np.where(thresh_gray == 0, orign[:, :, 0], 255)
    res_img[:, :, 1] = np.where(thresh_gray == 0, orign[:, :, 1], 255)
    res_img[:, :, 2] = np.where(thresh_gray == 0, orign[:, :, 2], 255)
    # # 进行图像对比度增强
    res_img = img_enhance(res_img)
    return res_img


def img_enhance(img):
    # 进行图像对比度增强
    image = Image.fromarray(img)
    enh_con = ImageEnhance.Contrast(image)
    image = enh_con.enhance(1.5)  # 参数越大，对比度越大
    enh_col = ImageEnhance.Color(image)
    image = enh_col.enhance(1.5)

    return np.asarray(image)


def crop_minAreaRect(img, rect):
    # rotate img
    angle = rect[2]
    rows, cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    img_rot = cv2.warpAffine(img, M, (cols, rows))

    # rotate bounding box
    rect0 = (rect[0], rect[1], 0.0)
    box = cv2.boxPoints(rect0)
    pts = np.int0(cv2.transform(np.array([box]), M))[0]
    pts[pts < 0] = 0

    # crop
    img_crop = img_rot[pts[1][1]:pts[0][1],
               pts[1][0]:pts[2][0]]

    return img_crop


if __name__ == '__main__':
    img_dir_path = './'
    img_list = glob.glob(os.path.join(img_dir_path, '*.jp*'))
    print('处理图片数量:{}'.format(len(img_list)))
    for image_path in img_list:
        generate_label_image(image_path)
