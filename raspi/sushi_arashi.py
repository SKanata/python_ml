#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import cv2
import pandas as pd
import numpy as np
import pyocr
import pyocr.builders
import time
import traceback
from picamera.array import PiRGBArray
from picamera import PiCamera
from PIL import Image
from sklearn.decomposition import RandomizedPCA
from sklearn.externals import joblib

STANDARD_SIZE = (400, 40)
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
        if contour.size > 700:
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

def extract_characters(img, clf, margin, tool):
    letter_boxes = detect_letters(img)

    # 抽出した文字領域を走査し、抽出対象領域を判定
    for num, box in enumerate(letter_boxes):
        # Caution: numpy syntax expects [y:y+h, x:x+w]
        img_trimed = img[box[1]-margin:box[1]+box[3]+margin, box[0]-margin:box[0]+box[2]+margin]

        # サイズを揃えるために一旦 PIL 形式に変換
        # モデルは STANDARD_SIZE で学習済み
        pil_img = Image.fromarray(np.uint8(img_trimed))
        pil_img_resized = pil_img.resize(STANDARD_SIZE)

        # np.array に戻す
        img_resized = np.asarray(pil_img_resized)

        # 一次元配列にする
        test_x = flatten_image(img_resized).astype(np.float64)
        
        prediction = clf.predict(test_x)
        #print(output_filename + "_____" + str(prediction)
        if prediction[0] == 1:
            break
        elif prediction[0] == 0:
            continue
        else:
            sys.exit(1)

    # 抽出した領域に対して OCR かける       
    img_gray = cv2.cvtColor(img_trimed, cv2.COLOR_RGB2GRAY)
    ret, img_binary = cv2.threshold(img_gray, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY) #+cv2.THRESH_OTSU)
    pil_img_binaly = Image.fromarray(np.uint8(img_binary))
    txt = tool.image_to_string( # ここでOCRの対象や言語，オプションを指定する
        pil_img_binaly,
        lang='eng',
        builder=pyocr.builders.TextBuilder()
    )
    print("あとはラズパイで'%(txt)s'と打つだけだ！" % {'txt': txt})
    

def main():
    # OCR のロード
    tools = pyocr.get_available_tools()
    if len(tools) == 0:
        print("No OCR tool found")
        sys.exit(1)
    tool = tools[0]


    # 学習済みのSVMモデル
    clf = joblib.load('model.pkl')
    margin = 5
    i = 0
    # initialize the camera and grab a reference to the raw camera capture
    camera = PiCamera()
    camera.resolution = (2592, 1952)
    #    camera.resolution = (3280, 2464)
    try:
        camera.start_preview()
        time.sleep(5)
        camera.stop_preview()
        a = input('Ready. Please press any key to start.')
        while True:
            with PiRGBArray(camera) as stream:
                camera.capture(stream, format='bgr')
                # At this point the image is available as stream.array
                img = stream.array
                #        img = frame.array
            cv2.imwrite('capture_%(i)s.png' % {'i': i}, img)
            extract_characters(img, clf, margin, tool)
            i += 1
    except:
        print(traceback.format_exc())

#    finally:
#        camera.stop_preview()

if __name__ == '__main__':
    main()
