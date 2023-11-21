import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
#########该函数能够读取磁盘中的图片文件，默认以彩色图像的方式进行读取
def imread_photo(filename, flags=cv2.IMREAD_COLOR):
    return cv2.imread(filename, flags)


##############这个函数的作用就是来调整图像的尺寸大小，当输入图像尺寸的宽度大于阈值（默认1000），我们会将图像按比例缩小#######
def resize_photo(imgArr,MAX_WIDTH = 1000):
    img = imgArr
    rows, cols= img.shape[:2]     #获取输入图像的高和宽即第0列和第1列
    if cols >  MAX_WIDTH:
        change_rate = MAX_WIDTH / cols
        img = cv2.resize(img ,( MAX_WIDTH ,int(rows * change_rate) ), interpolation = cv2.INTER_AREA)
    return img

# ################高斯平滑###############
#我们首先会对图像水平方向进行卷积，然后再对垂直方向进行卷积，其中sigma代表高斯卷积核的标准差
def gaussBlur(image,sigma,H,W,_boundary = 'fill', _fillvalue = 0):
    #水平方向上的高斯卷积核
    gaussKenrnel_x = cv2.getGaussianKernel(sigma,W,cv2.CV_64F)
    #进行转置
    gaussKenrnel_x = np.transpose(gaussKenrnel_x)
    #图像矩阵与水平高斯核卷积
    gaussBlur_x = signal.convolve2d(image,gaussKenrnel_x,mode='same',boundary=_boundary,fillvalue=_fillvalue)
    #构建垂直方向上的卷积核x
    gaussKenrnel_y = cv2.getGaussianKernel(sigma,H,cv2.CV_64F)
    #图像与垂直方向上的高斯核卷积核
    gaussBlur_xy = signal.convolve2d(gaussBlur_x,gaussKenrnel_y,mode='same',boundary= _boundary,fillvalue=_fillvalue)
    return gaussBlur_xy


def chose_licence_plate(contours, Min_Area=2000):
    temp_contours = []
    for contour in contours:
        if cv2.contourArea(contour) > Min_Area:
            temp_contours.append(contour)
    car_plate = []
    for temp_contour in temp_contours:
        rect_tupple = cv2.minAreaRect(temp_contour)
        rect_width, rect_height = rect_tupple[1]
        if rect_width < rect_height:
            rect_width, rect_height = rect_height, rect_width
        aspect_ratio = rect_width / rect_height
        # 车牌正常情况下宽高比在2 - 5.5之间
        if aspect_ratio > 2 and aspect_ratio < 5.5:
            car_plate.append(temp_contour)
            rect_vertices = cv2.boxPoints(rect_tupple)
            rect_vertices = np.intp(rect_vertices)
    return car_plate

def license_segment( car_plates ):
    if len(car_plates)==1:
        for car_plate in car_plates:
            row_min,col_min = np.min(car_plate[:,0,:],axis=0)
            row_max, col_max = np.max(car_plate[:, 0, :], axis=0)
            cv2.rectangle(img, (row_min,col_min), (row_max, col_max), (0,255,0), 2)
            card_img = img[col_min:col_max,row_min:row_max,:]
            cv2.imshow("img", img)
        cv2.imwrite( "card_img.png", card_img)
        cv2.imshow("card_img.png", card_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return  "card_img.png"

if __name__ == "__main__":

    img = imread_photo("image/3.jpg")  # 默认读取彩色图片
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将彩色图片转换为灰色图片,灰色图像更便于后续处理。
    rgray_img = resize_photo(gray_img)

    # 高斯平滑
    blurImage = gaussBlur(gray_img, 5, 400, 400, 'symm')
    #对bIurImage进行灰度级显示
    blurImage = np.round(blurImage)
    blurImage = blurImage.astype(np.uint8)

    kernel = np.ones((10, 10), np.uint8)
    #开运算
    img_opening = cv2.morphologyEx(blurImage, cv2.MORPH_OPEN, kernel)
    # cv2.imshow("GaussBlur_gray_img", blurImage)
    # cv2.imshow("xingtai_gray_img", img_opening)


    #将两幅图像合成为一幅图像
    img_opening = cv2.addWeighted(rgray_img, 1, img_opening, -1, 0)
    # cv2.imshow("hecheng_gray_img", img_opening)



    #阈值分割
    t, result_img = cv2.threshold(img_opening, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2.imshow("yuzhi_gray_img",result_img)
    #canny边缘检测
    img_edge = cv2.Canny(result_img, 100, 200)
    # cv2.imshow("bianyuan_gray_img", img_edge)
    #闭运算来填充白色物体内细小黑色空洞的区域并平滑其边界
    kernel1 = np.ones((18, 18), np.uint8)
    img_edge1 = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, kernel1)
    # cv2.imshow("biyunsuan", img_edge1)

    kernel2 = np.ones((10, 10), np.uint8)
    img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, kernel2)
    # cv2.imshow("kaiyunsuan", img_edge2)

    kernel = np.ones((20, 20), np.uint8)
    img_dilate = cv2.dilate(img_edge2, kernel)  # 膨胀
    cv2.imshow("dilate", img_dilate)  # 显示图片

    # #查找图像边缘整体形成的矩形区域, contours是找到的多个轮廓
    contours, hierarchy = cv2.findContours(img_dilate,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # cv2.imshow("xunzhao", hierarchy)

    draw_img = img.copy()
    result = cv2.drawContours(draw_img, contours, -1, (0, 0, 255), 2)
    # 画出带有轮廓的原始图片
    cv2.imshow('ret',result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    car_plates = chose_licence_plate(contours)
    card_img = license_segment(car_plates)
