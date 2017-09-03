#!/usr/bin/env python
import time
from PIL import Image
import numpy as np
import os
import pandas as pd
import pylab as pl
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import RandomizedPCA
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from plot_decision_regions import plot_decision_regions
import matplotlib.pyplot as plt

STANDARD_SIZE = (400, 40)

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
    start = time.time()
    img_dir = "output/"
    images = [img_dir + f for f in os.listdir(img_dir)]
    labels = ["ok" if "0020.png" in f.split('_')[-1] else "ng" for f in images]
    np.random.seed(10)
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
    
    pipe_svc = Pipeline(
        [
            ('scl', StandardScaler()),
            ('pca', (RandomizedPCA())),
            ('clf', SVC(random_state=0))
        ]
    )
#    param_range = [0.0001]
    param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    param_grid = [
        {"pca__n_components" : list(range(2, 6)), 'clf__C': param_range, 'clf__kernel': ['linear']},
        {"pca__n_components" : list(range(2, 6)), 'clf__C': param_range, 'clf__kernel': ['rbf'], 'clf__gamma': param_range}
    ]

    gs = GridSearchCV(
        estimator=pipe_svc,
        param_grid=param_grid,
        scoring='accuracy',
        cv=10,
        n_jobs=-1
    )

    gs = gs.fit(train_x, train_y)
    print(gs.best_score_)
    print(gs.best_params_)
    clf = gs.best_estimator_
    clf.fit(train_x, train_y)
    joblib.dump(clf, 'model.pkl')
    test_x, test_y = data[is_train == False], y[is_train == False]
    print(pd.crosstab(test_y, clf.predict(test_x), rownames=['Actual'], colnames=['Predicted']))
    
    """
    # training a classifier
    pca = RandomizedPCA(n_components=2)
    stdsc = StandardScaler()
    train_x = pca.fit_transform(train_x)
    train_x_std = stdsc.fit_transform(train_x)
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
    plt.show()
    print(pd.crosstab(test_y, svm.predict(test_x_std), rownames=['Actual'], colnames=['Predicted']))
"""    
    end = time.time()
    print('Elapsed time: %(time)s' % {'time': end - start})
if __name__ == '__main__':
    main()

