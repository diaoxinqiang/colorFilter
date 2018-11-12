import cv2
import numpy as np


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


# 红色的hsv值：0-10，146-180
lower_red = np.array([[0, 43, 46], [156, 43, 46]])
upper_red = np.array([[10, 255, 255], [180, 255, 255]])
# 读取图片
image = cv2.imread('red_paper/0.jpg', flags=cv2.IMREAD_COLOR)
# 图片如果太大，缩小四倍
image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
cv2.imshow("original", image)
# 过滤其他颜色，只留下红色
filter_image = leave_color(image, lower_red, upper_red)
cv2.imshow('filter_image', filter_image)
# 二值化 得到比较干净的红笔痕迹
_, threshold_image = cv2.threshold(filter_image, 1, 255, cv2.THRESH_BINARY)
cv2.imshow('threshold_image', threshold_image)
# 按q 退出
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print
        "I'm done"
        break
