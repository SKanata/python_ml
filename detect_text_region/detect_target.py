#!/usr/bin/env python
from PIL import Image
import numpy as np
import os
import pandas as pd
import pylab as pl
from sklearn.decomposition import RandomizedPCA
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from plot_decision_regions import plot_decision_regions
import matplotlib.pyplot as plt

STANDARD_SIZE = (300, 40)

def img_to_matrix(filename, verbose=False):
    """
    takes a filename and turns it into a numpy array of RGB pixels
    """
    img = Image.open(filename)
    if verbose==True:
        print("changing size from %s to %s" % (str(img.size), str(STANDARD_SIZE)))
    img = img.resize(STANDARD_SIZE)
    imgArray = np.asarray(img)
    # PIL は RGB だが、OpenCV は BGR。
    # OpenCVに合わせて BGR 変換してから return する
    imgArray = imgArray[:, :, ::-1] 
    return imgArray

def flatten_image(img):
    """
    takes in an (m, n) numpy array and flattens it 
    into an array of shape (1, m * n)
    """
    s = img.shape[0] * img.shape[1] * img.shape[2]
    img_wide = img.reshape(1, s)
    return img_wide[0]


def main():
    img_dir = "output/"
    images = [img_dir + f for f in os.listdir(img_dir)]
    labels = ["ok" if "0020.png" in f.split('_')[-1] else "ng" for f in images]
    print(images)
    print(labels)
    
    data = []
    for image in images:
        img = img_to_matrix(image)
        img = flatten_image(img)
        data.append(img)
    
    data = np.array(data)
    is_train = np.random.uniform(0, 1, len(data)) <= 0.7
    y = np.where(np.array(labels) == 'ok', 1, 0)
    train_x, train_y = data[is_train], y[is_train]
    
#    pca = RandomizedPCA(n_components=2)
#    X = pca.fit_transform(data)
#    df = pd.DataFrame({"x": X[:, 0], "y": X[:, 1], "label":np.where(y==1, "ok", "ng")})
#    colors = ["red", "yellow"]
#    for label, color in zip(df['label'].unique(), colors):
#        mask = df['label']==label
#        pl.scatter(df[mask]['x'], df[mask]['y'], c=color, label=label)
#    pl.legend()
#    pl.show()
#
    
    # training a classifier
    pca = RandomizedPCA(n_components=2)
    stdsc = StandardScaler()
    train_x = pca.fit_transform(train_x)
    train_x_std = stdsc.fit_transform(train_x)
#    train_x_std[:, 0] = (train_x[:, 0] - train_x[:, 0].mean()) / train_x[:, 0].std()
#    train_x_std[:, 1] = (train_x[:, 1] - train_x[:, 1].mean()) / train_x[:, 1].std()
    svm = SVC(kernel='rbf', random_state=0, gamma=10.0, C=5.0)
    svm.fit(train_x_std, train_y)
    print(train_x_std)
    print(train_y)
#    plot_decision_regions(train_x_std, train_y, classifier=svm)
    joblib.dump(svm, 'model.pkl')
    joblib.dump(pca, 'pca.pkl')
    joblib.dump(stdsc, 'stdsc.pkl')
    # evaluating the model
    test_x, test_y = data[is_train == False], y[is_train == False]
    test_x = pca.transform(test_x)
    test_x_std = stdsc.transform(test_x)
#    test_x_std[:, 0] = (test_x[:, 0] - test_x[:, 0].mean()) / test_x[:, 0].std()
#    test_x_std[:, 1] = (test_x[:, 1] - test_x[:, 1].mean()) / test_x[:, 1].std()
    plt.show()
    print(pd.crosstab(test_y, svm.predict(test_x_std), rownames=['Actual'], colnames=['Predicted']))
    
if __name__ == '__main__':
    main()

