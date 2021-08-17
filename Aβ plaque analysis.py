import os
import cv2
import shutil
import numpy as np
from tqdm import tqdm

# 返回一个cv2 image的RGB或HSV color均值, HSV取值范围为0~360, 0~1, 0~1
def ColorMean(im, mode): 
    if (mode == 'RGB') or (mode == 'rgb'): 
        b, g, r = cv2.split(im)
        return np.mean(np.array(r)), np.mean(np.array(g)), np.mean(np.array(b))
    else:
        if (mode == 'HSV') or (mode == 'hsv'): 
            im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(im)
            return 2*np.mean(np.array(h)), np.mean(np.array(s))/255, np.mean(np.array(v))/255
        else: print('Error: not a cv2 image type.')

# test = cv2.imread("TGh17-6M-20X-hip-2.tif")
# print(ColorMean(test, 'rgb'))

# TY's color balance，接收一个cv2 image，返回一个cv2 image
def TYCB(im): 
    stander = [223, 223, 233]
    b, g, r = cv2.split(im)
    r_mean, g_mean, b_mean = ColorMean(im, 'rgb')
    b_ = b/b_mean * stander[2]; b[b>255] = 255; b[b<0] = 0
    g_ = g/g_mean * stander[1]; g[g>255] = 255; g[g<0] = 0
    r_ = r/r_mean * stander[0]; r[r>255] = 255; r[r<0] = 0
    return cv2.merge([b_.astype(np.uint8), g_.astype(np.uint8), r_.astype(np.uint8)])

def AbetaPlaqueAnalysis(im): # 接受一个cv2 image，返回一个处理后的cv2 image
    try: im = TYCB(im)
    except: print("TYCB failed.")
    im_copy = np.copy(im)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(im)
    lower_threshold = np.array([0, 10, 10])
    upper_threshold = np.array([10, 190,190])
    mask = cv2.inRange(im, lower_threshold, upper_threshold)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=12)
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=12)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=8)
    mask = cv2.medianBlur(mask, 9)
    contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) # RETR_EXTERNAL 如果你选择这种模式的话，只会返回最外边的的轮廓，所有的子轮廓都会被忽略掉
    num = 0 # 斑块总数
    big = 0 # 大斑块总数
    mid = 0 # 中斑块总数
    small = 0 # 小斑块总数
    fitEllipseError = 0
    area_threshold2 = mask.size*(8000/12533760)
    area_threshold1 = mask.size*(3000/12533760)
    for i in range(len(contours)): 
        try:  
            ellipse = cv2.fitEllipse(contours[i]) # 寻找椭圆
            pt = (int(ellipse[0][0]),int(ellipse[0][1])) # 记录椭圆圆心坐标
            if ellipse[1][0]*ellipse[1][1] > area_threshold2: 
                big = big+ 1
                im_copy = cv2.ellipse(im_copy, ellipse,(0,0,255),2) # 在原始彩色图像上绘制椭圆
            else: 
                if area_threshold1 < ellipse[1][0]*ellipse[1][1] < area_threshold2: 
                    mid = mid + 1
                    im_copy = cv2.ellipse(im_copy, ellipse,(0,255,0),2) # 在原始彩色图像上绘制椭圆
                else: 
                    small = small + 1
                    im_copy = cv2.ellipse(im_copy, ellipse,(255,0,0),2) # 在原始彩色图像上绘制椭圆
            num = num + 1
        except: 
            fitEllipseError += 1
    stringtotal = "Abeta plaque:" + str(num)
    stringbig = "Big Abeta plaque:" + str(big)
    stringmid = "Mid Abeta plaque:" + str(mid)
    stringsmall = "Small Abeta plaque:" + str(small)
    cv2.putText(im_copy, stringtotal, (50,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 3)
    cv2.putText(im_copy, stringbig, (50,200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 3)
    cv2.putText(im_copy, stringmid, (50,300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 3)
    cv2.putText(im_copy, stringsmall, (50,400), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 3)
    return mask, im_copy, num, big, mid, small

# 给定输入和输出目录
filepath_input = "./Original images"
filepath_output = os.path.abspath(os.path.join(os.path.join(filepath_input, os.pardir), "Output"))

# 如果存在jpg目录则提前删除然后新建一个，如果不存在jpg目录则新建一个
if os.path.exists(filepath_output): 
    shutil.rmtree(filepath_output) 
    os.mkdir(filepath_output)
else: os.mkdir(filepath_output)

for root, dirs, files in os.walk(filepath_input, topdown=False): 
    # files.sort(key=lambda x:int(x[:-5])) # 根据文件名排序files中的文件顺序，这里结尾扩展名是.jpeg，所以是-5
    for name in tqdm(files):
        # If a jpg is present, alarming this.
        if os.path.isfile(os.path.splitext(os.path.join(filepath_output, name))[0] + ".jpg"):
            print("A jpg file already exists for %s" % name)
        else:
            outfile_name = os.path.splitext(os.path.join(filepath_output, name))[0] + ".jpg"
            mask_name = os.path.splitext(os.path.join(filepath_output, name))[0] + "_mask" + ".jpg"
            if os.path.join(root, name)[-9:] == ".DS_Store": pass
            else:
                if os.path.splitext(os.path.join(root, name))[1].lower() == ".tif" or ".jpg" or ".jpeg":
                    try:
                        im = cv2.imread(os.path.join(root, name)) # 读取图片
                        mask, mod, num, big, mid, small = AbetaPlaqueAnalysis(im) # Abeta plaque counting
                        cv2.imwrite(outfile_name, mod) # 保存处理后的图片
                        # cv2.imwrite(mask_name, mask) # 保存mask
                    except Exception:
                        print("cannot proccess")
                else: print("cannot proccess")