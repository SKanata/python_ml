#!/usr/bin/env python
import sys
import cv2
import pandas as pd
import numpy as np
import pyocr
import pyocr.builders
import time
from PIL import Image
from sklearn.decomposition import RandomizedPCA
from sklearn.externals import joblib

STANDARD_SIZE = (300, 40)
BINARY_THRESHOLD = 200

def detect_letters(img):
    """
    文字領域を抽出する関数。下記の c++ を移植
    https://stackoverflow.com/questions/23506105/extracting-text-opencv
    """
    
    # グレースケールに変換
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # エッジ検出
    img_sobel = cv2.Sobel(img_gray, cv2.CV_8U, 1, 0) #, 3, 1, 0, cv2.BORDER_DEFAULT)

    # 2値化
    ret, img_threshold = cv2.threshold(img_sobel, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # モルフォロジー処理
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(17, 3))
    img_morphology = cv2.morphologyEx(img_threshold, cv2.MORPH_CLOSE, kernel)

    # 輪郭抽出
    image, contours, hierarchy = cv2.findContours(img_morphology, 0, 1)
    # print(type(image))     -> ndarray
    # print(type(contours))  -> list
    # print(type(hierarchy)) -> ndarray

    # contour で文字領域を切り出す
    bound_rect = []
    for contour in contours:
        if contour.size > 1000:
            # rect => (x, y, width, height)
            rect = cv2.boundingRect(cv2.approxPolyDP(contour, 3, True))
            if rect[2]  > rect[3] * 2:
                bound_rect.append(rect)

    return bound_rect

def flatten_image(img):
    """
    takes in an (l, m, n) numpy array and flattens it 
    into an array of shape (1, l * m * n)
    """
    s = img.shape[0] * img.shape[1] * img.shape[2]
    img_wide = img.reshape(1, s)
    return img_wide

def main():
    start_time = time.time()
    # 学習済みのSVMモデル
    clf = joblib.load('model.pkl')

    read_time = time.time()
    img_path = sys.argv[1]
    margin = 5
    #img_path = 'test_f2.png'
    header = 'pysushi_'
    img = cv2.imread(img_path)
    letter_boxes = detect_letters(img)

    # 抽出した文字領域を走査し、抽出対象領域を判定
    for num, box in enumerate(letter_boxes):
        output_filename = "extract_tmp_output/%s%04d.png" % (header, num)

        # Caution: numpy syntax expects [y:y+h, x:x+w]
        img_trimed = img[box[1]-margin:box[1]+box[3]+margin, box[0]-margin:box[0]+box[2]+margin]
        cv2.imwrite(output_filename, img_trimed)

        # サイズを揃えるために一旦 PIL 形式に変換
        # モデルは STANDARD_SIZE で学習済み
        pil_img = Image.fromarray(np.uint8(img_trimed))
        pil_img_resized = pil_img.resize(STANDARD_SIZE)

        # np.array に戻す
        img_resized = np.asarray(pil_img_resized)
        #cv2.imshow(output_filename, img_resized)
        #cv2.waitKey()

        # 一次元配列にする
        test_x = flatten_image(img_resized)

        print(output_filename + "_____" + str(clf.predict(test_x)))

        # 抽出対象を検出したら終了
        if clf.predict(test_x)[0] == 1:
            break

    # 抽出した領域に対して OCR かける   
    tools = pyocr.get_available_tools()
    if len(tools) == 0:
        print("No OCR tool found")
        sys.exit(1)
    
    tool = tools[0]
    cv2.imshow('trimed', img_trimed)
    cv2.waitKey()

    img_gray = cv2.cvtColor(img_trimed, cv2.COLOR_RGB2GRAY)
    cv2.imwrite('01_gray_' + img_path, img_trimed)
    cv2.imshow('gray', img_gray)
    cv2.waitKey()
    
    ret, img_binary = cv2.threshold(img_gray, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)
    cv2.imshow('binary', img_binary)
    cv2.waitKey()
    cv2.imwrite('02_binary_%(thresh)s_%(img_path)s' % {'thresh': BINARY_THRESHOLD, 'img_path': img_path}, img_binary) 

    # モルフォロジー処理
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2, 1))
    img_binary = cv2.erode(img_binary,kernel,iterations = 1) 
#    img_binary = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite('03_morpho_%(thresh)s_%(img_path)s' % {'thresh': BINARY_THRESHOLD, 'img_path': img_path}, img_binary) 
    cv2.imshow('thresh', img_binary)
    cv2.waitKey()
    pil_img_binaly = Image.fromarray(np.uint8(img_binary))
    txt = tool.image_to_string( # ここでOCRの対象や言語，オプションを指定する
        pil_img_binaly,
        lang='eng',
        builder=pyocr.builders.TextBuilder()
    )
    print("あとはラズパイで'%(txt)s'と打つだけだ！" % {'txt': txt})
    #    print(txt)
    end_time = time.time()
    print("load pkl time: %(load_pkl)s" % {'load_pkl': read_time - start_time})
    print("detect time: %(detect)s" % {'detect': end_time - read_time })
    print("total time: %(total)s" % {'total': end_time - start_time })
    
    
if __name__ == '__main__':
    main()
