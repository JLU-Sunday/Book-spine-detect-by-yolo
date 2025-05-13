# -*- coding: utf-8 -*-
# import paddlehub as hub
import cv2
import numpy as np
import imutils
import math
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import copy
import tools
import time
import os
import sys

import cv2
import numpy as np


def cv_imread(file_path):
    """
    读取图片并修改分辨率（可选）

    :param file_path: 图片路径
    :param target_size: 目标分辨率 (宽, 高)，例如 (640, 480)。如果为 None，则在图像宽 > 1920 或高 > 1080 时按比例缩小。
    :return: 读取并调整分辨率后的图片（BGR格式）
    """
    # 读取图像
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)

    # 检查是否成功读取图像
    if cv_img is None:
        raise ValueError(f"无法读取图片: {file_path}")

    # 获取图像原始高度和宽度
    h, w = cv_img.shape[:2]



    # 当 target_size 为 None 时，检查是否需要缩小
    if w > 1920 or h > 1080:
        # 计算缩放比例，取宽度和高度缩放因子中最小的那个
        scale = min(1920/ w, 1280 / h)
        # 计算新的宽度和高度
        new_w = int(w * scale)
        new_h = int(h * scale)
        # 按比例缩小图像
        cv_img = cv2.resize(cv_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return cv_img


def linequal(j, k):
    dis = 0
    dis += dist.euclidean((j[0], j[1]), (k[0], k[1]))
    dis += dist.euclidean((j[2], j[3]), (k[2], k[3]))
    if dis < 0.01:
        return True
    else:
        return False


def merge_line(lst, line):
    flag = False
    for l in lst:
        # print('l[0]:',l[0])
        # print('line:',line)
        if linequal(l[0], line[0]):
            flag = True
    if flag == False:
        lst.append(line)
    return lst


def merge(fitlines, lines):
    flag = False
    for i, ls in enumerate(fitlines):
        for line in ls:
            for l in lines:
                # print('l[0]:',l[0])
                # print('line[0]:',line[0])
                if linequal(l[0], line[0]):
                    flag = True
                    for ll in lines:
                        fitlines[i] = merge_line(fitlines[i], ll)
                        return fitlines
    if flag == False:
        fitlines.append(lines)
        return fitlines


def crop(img, rotated_box):
    center, size, angle = rotated_box[0], rotated_box[1], rotated_box[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))

    # print(angle)

    if size[0] < size[1]:

        w = size[0]
        h = size[1]
    else:
        angle -= 90
        w = size[1]
        h = size[0]
    size = (w, h)

    height, width = img.shape[0], img.shape[1]

    M = cv2.getRotationMatrix2D(center, angle, 1)
    img_rot = cv2.warpAffine(img, M, (width, height))
    img_crop = cv2.getRectSubPix(img_rot, size, center)

    return img_crop
    # show([img, img_rot, img_crop])


def getDis(pointX, pointY, lineX1, lineY1, lineX2, lineY2):
    a = lineY2 - lineY1
    b = lineX1 - lineX2
    c = lineX2 * lineY1 - lineX1 * lineY2
    dis = (math.fabs(a * pointX + b * pointY + c)) / (math.sqrt(a * a + b * b))
    return dis

def rotatePoint(xc, yc, xp, yp, theta):
    xoff = xp - xc;
    yoff = yp - yc;
    cosTheta = math.cos(theta)
    sinTheta = math.sin(theta)
    pResx = cosTheta * xoff + sinTheta * yoff
    pResy = - sinTheta * xoff + cosTheta * yoff
    return str(int(xc + pResx)), str(int(yc + pResy))
def computeCrossX(y, value):
    x1, y1, x2, y2 = value
    if abs(y1 - y2) < 0.001:
        return (y1 + y2) / 2
    else:
        x = (x1 - x2) * (y - y1) / (y1 - y2) + x1
        return x


import paddlehub as hub

ocr = hub.Module(name="chinese_ocr_db_crnn_server")
i_i = 1
    # 遍历文件夹中的所有文件
    #the way of pictures
directory_path = "E:\BaiduNetdiskDownload\复现\Book_Dataset_1\Dataset_1"

for filename in os.listdir(directory_path):
    file_path = os.path.join(directory_path, filename)
    # file_path = "D:\\Users\\11256\\Desktop\\labeling_img\\img1\\微信图片_20240502134156.jpg"
    print(file_path)

    img = cv_imread(file_path)
    picture_to_show = copy.deepcopy(img)
    size = img.shape
    w = size[1]  # 宽度
    h = size[0]  # 高度
    print('w:', w, 'h:', h)
    theta_step = 1


    #thetas = [i for i in range(-90, 90, theta_step)]
    rmax = int(math.hypot(h, w))
    #hough_space = np.zeros((len(thetas), 2 * rmax + 1), dtype=np.uint16)

    Length_thd = 15  # 线检测的长度阈值，小于该阈值的线段被丢弃
    Dis_thd = 3  # 线检测的距离阈值
    AngleSinThd = 0.8  # 9999对线检测的结果进行滤波，水平线角度为pi/2，角度正弦值大于AngleSinThd或小于-AngleSinThd的线段被丢弃
    MyDisThd = 5 # 线段l的两个端点距离拟合直线小于该阈值，将l加入到拟合直线组中
    NumOfPeaks = 120
    LineRatio = 0.75
    BSpineLen = 0.65
    NumAngle = 30
    v_sin_max = 0
    v_sin_min = 0
    '''
    img = np.zeros((600, 600,3), dtype=np.uint8)
    cv2.line(img, (10, 10), (20, 20), (255,255,255), 1)
    cv2.line(img, (30, 30), (50, 50), (255,255,255), 1)
    cv2.line(img, (60, 60), (80, 80), (255,255,255), 1)
    cv2.line(img, (100, 100), (120, 120), (255,255,255), 1)
    #cv2.line(edge, (300, 300), (400, 400), 255, 1)
    '''
    # plt.imshow(img)
    # plt.show()

    # reader = easyocr.Reader(['en'], gpu=False)
    # result = reader.readtext(img)
    # result
    '''
    sharpen_op = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharpen_image = cv2.filter2D(img, cv2.CV_32F, sharpen_op)
    img = cv2.convertScaleAbs(sharpen_image)
    '''
    start = time.time()

    # blur_img = cv2.GaussianBlur(img, (0, 0), 3)
    # img = cv2.addWeighted(img, 1.5, blur_img, -0.5, 0)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Create default Fast Line Detector (FSD)
    fld = cv2.ximgproc.createFastLineDetector(length_threshold=Length_thd, distance_threshold=Dis_thd,
                                              do_merge=True)
    # Detect lines in the image 获取所有线条
    lines = fld.detect(gray)

    end = time.time()

    print('高斯变换与线检测耗时：', str(end - start))
    image = fld.drawSegments(gray, lines)#检测到的线绘制出来
    # cv2.imshow("origin", image)
    # cv2.waitKey()
    # plt.figure("origin")
    # plt.imshow(image)
    # plt.show()
    # print(type(lines))
    # print("lines:")
    # print(lines)

    LineCategory = dict()
    print("number of lines:", len(lines))
    count = 0
    for l in lines:
        # print(l[0])
        for x1, y1, x2, y2 in l:

            D = dist.euclidean((x1, y1), (x2, y2))  # 欧式距离计算
            # print("D",D)
            if D == 0:
                continue
            # 如果y距离很小，认为垂直y轴线段，角度pi/2
            if abs(y2 - y1) < 0.001:
                rTheta = math.pi / 2
                ro = round((y1 + y2) / 2)
            else:
                slope = -(x2 - x1) / (y2 - y1)
                rTheta = math.atan(slope)
                ro = round((math.cos(rTheta) * x1 + math.sin(rTheta) * y1))

            Vsin = math.sin(rTheta)
            # if 1329<l[0][0]<1330:
            #    x=1
            if Vsin < -AngleSinThd or Vsin > AngleSinThd:
                count += 1
                continue

            '''角度滤波AngleThd'''

            '''角度滤波AngleThd'''
            # print("old", Vsin, NumAngle)
            # print(l)
            Vsin = round((Vsin + 1) * NumAngle) + 1
            # print("new", Vsin)
            ro = round((ro + rmax))
            '''if(Vsin,ro)==(56,4940):
                x=1
            if 763.7<x1<763.73:
                D=0
            '''
            if ro <= 2 * rmax:
                if v_sin_max<Vsin:
                    v_sin_max = Vsin
                if v_sin_min >Vsin:
                    v_sin_min = Vsin
    thetas = [i for i in range(v_sin_min, v_sin_max+1, theta_step)]
    hough_space = np.zeros((len(thetas), 2 * rmax + 1), dtype=np.uint16)
    # print(type(lines))
    start = time.time()
    #l 格式 x1 y1 x2 y2
    for l in lines:
        # print(l[0])
        for x1, y1, x2, y2 in l:

            D = dist.euclidean((x1, y1), (x2, y2))#欧式距离计算
            # print("D",D)
            if D == 0:
                continue
            #如果y距离很小，认为垂直y轴线段，角度pi/2
            if abs(y2 - y1) < 0.001:
                rTheta = math.pi / 2
                ro = round((y1 + y2) / 2)
            else:
                slope = -(x2 - x1) / (y2 - y1)
                rTheta = math.atan(slope)
                ro = round((math.cos(rTheta) * x1 + math.sin(rTheta) * y1))

            Vsin = math.sin(rTheta)
            # if 1329<l[0][0]<1330:
            #    x=1
            if Vsin < -AngleSinThd or Vsin > AngleSinThd:
                count += 1
                continue

            '''角度滤波AngleThd'''

            '''角度滤波AngleThd'''
            #print("old", Vsin, NumAngle)
            #print(l)
            Vsin = round((Vsin + 1) * NumAngle) + 1
            #print("new", Vsin)
            ro = round((ro + rmax))
            '''if(Vsin,ro)==(56,4940):
                x=1
            if 763.7<x1<763.73:
                D=0
            '''
            if ro <= 2 * rmax:
                if (Vsin, ro) in LineCategory:
                    LineCategory[(Vsin, ro)].append(l)
                else:
                    LineCategory[(Vsin, ro)] = [l]
                hough_space[Vsin, ro] += D
    # 打印非零元素的索引和值

    nonzero_indices = np.nonzero(hough_space)  # 获取非零元素的索引
    nonzero_values = hough_space[nonzero_indices]  # 非零元素对应的值

    # 显示索引和对应的值
    print("非零元素索引及值：")
    for (Vsin, ro), val in zip(zip(*nonzero_indices), nonzero_values):
        print(f"(Vsin={Vsin}, ro={ro}) -> D={val}")
    print('number of LineCategory:', len(LineCategory))
    print('discarded:', count)
    end = time.time()

    print('霍夫空间计算耗时：', str(end - start))
    #for key, value in LineCategory.items():
        #print('LineCategory:', key, value)

    start = time.time()

    num = len(LineCategory)
    #找到最大几个点的索引
    idx = np.argpartition(hough_space.ravel(), hough_space.size - NumOfPeaks)[-NumOfPeaks:]
    #转化为二维坐标
    Peaks = np.column_stack(np.unravel_index(idx, hough_space.shape))
    test = []
    for peak in Peaks:
        # 如果Hough空间内有这段距离 就在test中包括这个线段
        if hough_space[peak[0], peak[1]] != 0:
            test.append([peak[0], peak[1]])
       # print("peask", hough_space[peak[0], peak[1]])
    test = np.array(test)

    print(len(Peaks))
    peaks = test
    #print(peaks)
    print('len of peaks:', len(peaks))
    end = time.time()

    print('霍夫空间排序耗时：', str(end - start))
    # plt.imshow(hough_space)
    # plt.show()

    start = time.time()

    sortedlines = sorted(lines, key=lambda ls: ls[0][0])
    # filtered=filter(lambda l: l[0][0]<282, sortedlines)
    # print('filtered:',list(filtered))
    #print('type', type(peaks))
    for p in peaks:
        (Vsin, ro) = (p[0], p[1])

        D = hough_space[Vsin, ro]
        #还原Vsin
        OVsin = (Vsin - 1) / NumAngle - 1
        '''OVcos=math.sqrt(1-OVsin**2)
        if abs(OVcos)<0.01:
            filterThr=2*MyDisThd
        else:
            filterThr=2*abs(D*OVsin/OVcos)+2*MyDisThd'''
        filterThr = 4 * abs(D * OVsin) + 4 * MyDisThd
        # print('OVsin:',OVsin,'filterThr:',filterThr)
        # if (Vsin,ro)==(56,4940):
        #    xxx=0
        #print("Vsin,ro", Vsin, ro)
        value = LineCategory[(Vsin, ro)]
        #    print('ok?',value)
        for line1x, line1y, line2x, line2y in value[0]:
            # print(line1x,line1y,line2x,line2y)

            mx2 = (line1x + line2x) / 2
            my2 = (line1y + line2y) / 2
            filtered = filter(lambda l: mx2 - filterThr < l[0][0] < mx2 + filterThr, sortedlines)
            for l in filtered:
                for x1, y1, x2, y2 in l:
                    '''角度滤波AngleThd'''
                    if abs(y2 - y1) < 0.001:
                        rTheta = math.pi / 2
                    else:
                        slope = -(x2 - x1) / (y2 - y1)
                        rTheta = math.atan(slope)
                    fVsin = math.sin(rTheta)
                    if fVsin < -AngleSinThd or fVsin > AngleSinThd:
                        continue
                    '''角度滤波AngleThd'''
                    mx1 = (x1 + x2) / 2
                    my1 = (y1 + y2) / 2
                    #中心间距离
                    md = dist.euclidean((mx1, my1), (mx2, my2))
                    #第二段直线长度
                    dxy = dist.euclidean((x1, y1), (x2, y2))
                    dline = dist.euclidean((line1x, line1y), (line2x, line2y))
                    #分别计算候选线段的两个端点到当前直线线段的距离 dis1 和 dis2。
                    dis1 = getDis(x1, y1, line1x, line1y, line2x, line2y)
                    dis2 = getDis(x2, y2, line1x, line1y, line2x, line2y)
                    # if (dis1+dis2)/2<MyDisThd and md<=4*(dxy+dline)/7:
                    if (dis1 + dis2) / 2 < MyDisThd:
                        LineCategory[(Vsin, ro)].append(l)
                        # hough_space[Vsin,ro]+=D
                        break
    end = time.time()
    print('加入直线组耗时：', str(end - start))

    start = time.time()
    # 合并直线组，将有重复线段的直线组合并为一组
    fitlines = []
    for p in peaks:
        (Vsin, ro) = (p[0], p[1])
        if hough_space[Vsin, ro] == 0:
            continue

        value = LineCategory[(Vsin, ro)]
        merge(fitlines, value)

    # print('type of fitlines:',type(fitlines))
    # print(fitlines[0])
    print('length of fitlines:', len(fitlines))
    # print('fitlines:',fitlines)
    end = time.time()
    print('合并耗时：', str(end - start))

    start = time.time()
    NewLineCategory = dict()
    Lenlist = []
    for lines in fitlines:
        if len(lines) > 0:
            dis = 0

            minpx = 0
            minpy = 0
            maxpx = 0
            maxpy = 0
            plst = []
            for l in lines:
                for x1, y1, x2, y2 in l:
                    plst.append([x1, y1, x1 ** 2 + y1 ** 2])
                    plst.append([x2, y2, x2 ** 2 + y2 ** 2])

                    # compute the Euclidean distance between the two points,
                    D = dist.euclidean((x1, y1), (x2, y2))
                    dis = dis + D
            plst.sort(key=lambda l: l[2])

            minx = plst[0][0]
            if minx < 0:
                minx = 0
            miny = plst[0][1]

            maxx = plst[-1][0]
            if maxx < 0:
                maxx = 0
            maxy = plst[-1][1]

            maxD = dist.euclidean((minx, miny), (maxx, maxy))

            loc = [[minx, miny], [maxx, maxy]]
            loc = np.array(loc)
            fittedline = cv2.fitLine(loc, cv2.DIST_L2, 0, 0.01, 0.01)
            # print(type(rTheta),rTheta)
            rTheta = math.atan(-fittedline[0] / fittedline[1])
            NewVsin = math.sin(rTheta)  # 不再用霍夫空间表示法了
            Newro = math.cos(rTheta) * fittedline[2][0] + math.sin(rTheta) * fittedline[3][0]

            # break
            if dis / maxD > LineRatio:
                # print(len(value),"ok")
                Lenlist.append(maxD)
                lineMax = np.array([minx, miny, maxx, maxy])
                NewLineCategory[(NewVsin, Newro)] = lineMax

    print('len of NewLineCategory:', len(NewLineCategory))
    Lenlist.sort()

    NewLines = dict()
    for key, value in NewLineCategory.items():
        x1, y1, x2, y2 = value
        d = dist.euclidean((x1, y1), (x2, y2))
        if d / Lenlist[-1] > BSpineLen:
            # 以拟合直线的最大长度的百分比滤除直线，会存在最大长度特别长的情况，比如书架的两侧边缘，
            # 后期可以用文字框筛选掉书架边缘，再进行分割
            NewLines[key] = value

    end = time.time()
    print('拟合NewLines耗时：', str(end - start))

    print('len of NewLines:', len(NewLines))

    '''
    old=[]
    for key,value in NewLines.items():
        #print ('ok',value)
        if len(old) > 0:
            x1,y1,x2,y2 = old
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 4)  

        x1,y1,x2,y2 = value        
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 4) 
        #print(key)

        old=NewLines[key]   
    #cv2.line(img, (int(0), int(1041.1881)), (int(16.798567), int(1085.2413)), (255,0,0), 4) 
    plt.imshow(img)
    plt.show()
    '''
    start = time.time()
    FinalLines = NewLines.copy()
    for key, value in NewLines.items():
        for k, v in NewLines.items():
            if (key, value) == (k, v):
                continue
            '''
            mx1=(value[0]+value[2])/2
            my1=(value[1]+value[3])/2
            mx2=(v[0]+v[2])/2
            my2=(v[1]+v[3])/2               
            md=dist.euclidean((mx1, my1), (mx2, my2))
            dxy=dist.euclidean((value[0],value[1]), (value[2],value[3]))
            dline=dist.euclidean((v[0],v[1]), (v[2],v[3]))
            '''
            dis1 = getDis(value[0], value[1], v[0], v[1], v[2], v[3])
            dis2 = getDis(value[2], value[3], v[0], v[1], v[2], v[3])
            if (dis1 + dis2) / 2 < MyDisThd:
                dvalue = dist.euclidean((value[0], value[1]), (value[2], value[3]))
                dv = dist.euclidean((v[0], v[1]), (v[2], v[3]))
                # print(type(dvalue))
                # 文字识别弄好以后，用文字识别先滤除一遍直线，然后再删掉重复的
                # 特别倾斜的，也要滤除一次
                if dvalue - dv > 0.01:
                    if k in FinalLines.keys():
                        FinalLines.pop(k)
                else:
                    if key in FinalLines.keys():
                        FinalLines.pop(key)

    print('len of FinalLines:', len(FinalLines))
    end = time.time()
    print('拟合去重耗时：', str(end - start))

    '''
    old=[]
    for key,value in FinalLines.items():
        #print ('ok',value)
        if len(old) > 0:
            x1,y1,x2,y2 = old
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 4)  

        x1,y1,x2,y2 = value        
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 4) 

        old=FinalLines[key]   
    plt.imshow(img)
    plt.show()
    '''
    # keys = sorted(FinalLines,key=lambda item:item[0])
    # new_dict = sorted(FinalLines.items(),  key=lambda d: d[1][2])
    # 排序还需加强
    y = h / 2
    for key, value in FinalLines.items():
        # print(type(value))
        x = computeCrossX(y, value)
        FinalLines[key] = [value, x]
    new_dict = sorted(FinalLines.items(), key=lambda d: d[1][1])
    # print("hello:",new_dict)
    # print(keys)
    books_to_dota = []
    Index = 0
    for i in range(len(new_dict) - 1):
        # print(type(new_dict[i][1]))
        # print(new_dict[i+1][1])
        line1 = new_dict[i][1][0]
        line2 = new_dict[i + 1][1][0]
        (Vsin1, _) = new_dict[i][0]
        (Vsin2, _) = new_dict[i + 1][0]
        rTheta1 = math.asin(Vsin1)
        rTheta2 = math.asin(Vsin2)
        dTheta1 = math.degrees(rTheta1)
        dTheta2 = math.degrees(rTheta2)
        # print('dTheta1:',dTheta1)
        # print('dTheta2:',dTheta2)
        if abs(dTheta1 - dTheta2) > 3.5:
            continue

        Index += 1
        cnt = [(line1[0], line1[1]), (line1[2], line1[3]), (line2[0], line2[1]), (line2[2], line2[3])]
        '''print('cnt:',cnt)
        print('cnt:',cnt[0])
        cnt = np.int0(cnt)
        cv2.circle(img, (cnt[0][0], cnt[0][1]), 4, (255,255,255), 0)
        cv2.circle(img, (cnt[1][0], cnt[1][1]), 4, (255,255,255), 0)
        cv2.circle(img, (cnt[2][0], cnt[2][1]), 4, (255,255,255), 0)
        cv2.circle(img, (cnt[3][0], cnt[3][1]), 4, (255,255,255), 0)
        cv2.imshow("",img)
        cv2.waitKey()
        '''
        cnt = np.int0(cnt)
        cnt = np.array(cnt)
        rotated_box = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rotated_box)
        box = np.int0(box)

        # plt.imshow(img)
        # plt.show()
        # cv2.imshow("",img)
        # cv2.waitKey()
        # 这里，存储4个顶点还是存储box呢？

        left_point_x = np.min(box[:, 0])
        right_point_x = np.max(box[:, 0])
        top_point_y = np.min(box[:, 1])
        bottom_point_y = np.max(box[:, 1])

        left_point_y = box[:, 1][np.where(box[:, 0] == left_point_x)][0]
        right_point_y = box[:, 1][np.where(box[:, 0] == right_point_x)][0]
        top_point_x = box[:, 0][np.where(box[:, 1] == top_point_y)][0]
        bottom_point_x = box[:, 0][np.where(box[:, 1] == bottom_point_y)][0]
        # 上下左右四个点坐标(shun shi zhen)
        vertices = np.array(
            [[top_point_x, top_point_y], [right_point_x, right_point_y], [bottom_point_x, bottom_point_y],
             [left_point_x, left_point_y]])
        books_to_dota.append(vertices)


    points = books_to_dota


    # Global variables for tracking selected point
    selected_index = -1
    selected_number = -1
    #print("22223")
    #print("1",selected_index,selected_number)
    dota = []

    # Function to calculate distance between two points
    def distance(p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    # Mouse callback function
    def move_point(event, x, y, flags, param):
        global books_to_dota, selected_index, selected_number

        if event == cv2.EVENT_LBUTTONDOWN:
            # Find if a point is clicked
            for number, points_array in enumerate(books_to_dota):
                for index, point in enumerate(points_array):
                    if distance((x, y), point) < 10:  # Check distance tolerance (10 pixels)
                        selected_number = number
                        selected_index = index
                        break

        if event == cv2.EVENT_RBUTTONDOWN:
            # Find if a point is clicked
            for number, points_array in enumerate(books_to_dota):
                for index, point in enumerate(points_array):
                    if distance((x, y), point) < 10:  # Check distance tolerance (10 pixels)
                        selected_number = number
                        selected_index = index
                        books_to_dota = np.delete(books_to_dota, selected_number, axis=0)
                        break
        elif event == cv2.EVENT_MOUSEMOVE:
            # If a point is selected, update its position
            if selected_index != -1:
                books_to_dota[selected_number][selected_index] = [x, y]

        elif event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP:
            # Reset selected_index when mouse button is released
            selected_index = -1

    # Create a black image and a window
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', move_point)
    # 这行代码假设 picture_to_show 已经被定义
    real_img = copy.deepcopy(picture_to_show)

    while True:
        picture_to_show = copy.deepcopy(real_img)
        for number,points_array in enumerate(books_to_dota):
            for point in points_array:
                if number %2:
                    cv2.circle(picture_to_show, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)
                else:
                    cv2.circle(picture_to_show, (int(point[0]), int(point[1])), 5, (0, 0,255), -1)
        for  number, points_array in enumerate(books_to_dota):
            rect_points = points_array.reshape((-1, 1, 2))
            rect = cv2.minAreaRect(rect_points.astype(np.float32))
            #print("rect",rect)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            #print(box[0],box[3],box[2],box[1])
            if number %2:
                cv2.drawContours(picture_to_show,[box], 0, (0, 255, 0), 2)
            else:
                cv2.drawContours(picture_to_show, [box], 0, (0, 0, 255), 2)
        cv2.imshow('image', picture_to_show)
        if cv2.waitKey(1) & 0xFF == 27:
            selected_index = -1
            selected_number = -1
            books_to_dota_new = copy.deepcopy(books_to_dota)
            books_to_dota = []
            break

    cv2.destroyAllWindows()
    #print("Final Points:")
    #print(books_to_dota_new)
    for points_array in books_to_dota_new:
        rect_points = points_array.reshape((-1, 1, 2))
        rect = cv2.minAreaRect(rect_points.astype(np.float32))
        #print(rect)
        box = cv2.boxPoints(rect)
        box = box.tolist()
        min_y1 = min(box, key=lambda point: point[1])
        box.remove(min_y1)
        min_y2 = min(box, key=lambda point: point[1])

        # Finding the two points with the largest y-coordinates
        max_y1 = max(box, key=lambda point: point[1])
        box.remove(max_y1)
        max_y2 = max(box, key=lambda point: point[1])

        # Determining r1, r2, r3, r4 based on the definitions provided
        r1 = max(min_y1, min_y2, key=lambda point: point[0])
        r2 = max(max_y1, max_y2, key=lambda point: point[0])
        r3 = min(max_y1, max_y2, key=lambda point: point[0])
        r4 = min(min_y1, min_y2, key=lambda point: point[0])

        dota.append([r1,r2,r3,r4])
    dota = np.array(dota, dtype=int)

    # Reshape dota to have 8 columns
    dota = dota.reshape(len(dota), 8)

    # Print dota array
    #print('dota', dota)

    # Create new_columns array
    new_columns = np.array([['book', 0] for _ in range(len(dota))], dtype=object)

    # 将新列与原始数组合并
    if new_columns.size > 0:
        e = np.hstack((dota, new_columns))

        #print("Extended array:")
        #print(dota)
        e_str = np.array(e, dtype=str)
        name = filename.replace('.jpg','')
        #print(type(name))
        if not os.path.exists('C:/Users/11256\Desktop\pythonProject\q12341'):
            os.makedirs('C:/Users/11256\Desktop\pythonProject\q12341')
        # 将字符串数组保存到文本文件中
        np.savetxt('C:/Users/11256\Desktop\pythonProject\q12341/%s.txt'%name, e_str, fmt='%s', delimiter=' ')

        print("Array saved to output.txt")

# 指定文件夹路径



filename = 'only_one.jpg'



# plt.imshow(cutimg)
# plt.show()
# print(rotated_box)
# break

# cv2.line(bimg, (10, 10), (11, 11), (255,255,255), 1)
# cv2.line(bimg, (12, 12), (13, 13), (255,255,255), 1)
# cv2.line(bimg, (80, 80), (90, 90), (255,255,255), 1)
'''
for l in lines:
    for x1,y1,x2,y2 in l:
        #print(type(x1))
        cv2.line(bimg, (int(x1), int(y1)), (int(x2), int(y2)), (255,255,255), 1)
'''
