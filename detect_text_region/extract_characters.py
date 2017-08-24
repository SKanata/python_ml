#!/usr/bin/env python
import cv2
import pandas as pd
import numpy as np
import pyocr
import pyocr.builders
from PIL import Image
from sklearn.decomposition import RandomizedPCA
from sklearn.externals import joblib

STANDARD_SIZE = (300, 40)

def detect_letters(img):
    # グレースケールに変換
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # エッジ検出
    img_sobel = cv2.Sobel(img_gray, cv2.CV_8U, 1, 0) #, 3, 1, 0, cv2.BORDER_DEFAULT)

    # 2値化
    ret, img_threshold = cv2.threshold(img_sobel, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # モルフォロジー処理
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(18, 18))
    img_morphology = cv2.morphologyEx(img_threshold, cv2.MORPH_CLOSE, kernel)

    # 輪郭抽出
    image, contours, hierarchy = cv2.findContours(img_morphology, 0, 1)
    # print(type(image))     -> ndarray
    # print(type(contours))  -> list
    # print(type(hierarchy)) -> ndarray

    # conrour で文字領域を切り出す
    bound_rect = []
    for contour in contours:
        if contour.size > 100:
            # rect => (x, y, width, height)
            rect = cv2.boundingRect(cv2.approxPolyDP(contour, 3, True))
            if rect[2] > rect[3]:
                bound_rect.append(rect)

    return bound_rect

def flatten_image(img):
    """
    takes in an (m, n) numpy array and flattens it 
    into an array of shape (1, m * n)
    """
    s = img.shape[0] * img.shape[1] * img.shape[2]
    img_wide = img.reshape(1, s)
    return img_wide

def main():
    svm = joblib.load('model.pkl')
    pca = joblib.load('pca.pkl')
    stdsc = joblib.load('stdsc.pkl')
    img_path = 'test_f2.png'
    header = 'pysushi_'
    img = cv2.imread(img_path)
    letter_boxes = detect_letters(img)

    for num, box in enumerate(letter_boxes):
        output_filename = "output/%s%04d.png" % (header, num)

        # Caution: numpy syntax expects [y:y+h, x:x+w]
        img_trimed = img[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
        #cv2.imwrite(output_filename, img_trimed)
        pil_img = Image.fromarray(np.uint8(img_trimed))
        pil_img_resized = pil_img.resize(STANDARD_SIZE)
        img_resized = np.asarray(pil_img_resized)
        #cv2.imshow(output_filename, img_resized)
        #cv2.waitKey()
        img_wide = flatten_image(img_resized)
        test_x = pca.transform(img_wide)

        # 標準化しないとメモリ溢れてプログラムが終了する
        test_x_std = stdsc.transform(test_x)
        print(output_filename + "_____" + str(svm.predict(test_x_std)))
        if svm.predict(test_x_std)[0] == 1:
            break
        
    tools = pyocr.get_available_tools()
    if len(tools) == 0:
        print("No OCR tool found")
        sys.exit(1)
    
    tool = tools[0]
    
    txt = tool.image_to_string( # ここでOCRの対象や言語，オプションを指定する
        pil_img,
        lang='eng',
        builder=pyocr.builders.TextBuilder()
    )
    print(txt)

if __name__ == '__main__':
    main()
