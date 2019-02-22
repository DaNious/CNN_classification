import os
import random
import numpy as np
from skimage import io, color, transform


# read data from local
def import_data(train_size):
    train_x, train_y = [], []
    test_x, test_y = [], []

    path = os.getcwd() + '/MUS_DATASET_TIMESERIES1_IMG'
    folder_list = os.listdir(path)
    for fo in folder_list:
        #label_path = os.getcwd() + '/MUS_DATASET_IMG'
        label_data = fo
        pics_path = path + '/' + fo
        pics = os.listdir(pics_path)
        train_set = random.sample(pics, train_size)
        for pic in train_set:
            pics.remove(pic)
        test_set = pics
        for pic in train_set:
            pic_path = pics_path + '/' + pic
            img = io.imread(pic_path).astype(np.double)
            img = color.rgb2gray(img)
            img = transform.resize(img, (32, 32), mode='constant')
            train_x.append(img.reshape(-1))
            lab = pic.split('.')[0]
            lab = lab.split('_')[1]
            lab = lab[0:2]
            label = int(lab)
            if label == 4:
                label = [1, 0, 0, 0, 0]
            elif label == 9:
                label = [0, 1, 0, 0, 0]
            elif label == 11:
                label = [0, 0, 1, 0, 0]
            elif label == 14:
                label = [0, 0, 0, 1, 0]
            elif label == 16:
                label = [0, 0, 0, 0, 1]
            train_y.append(label)
        for pic in test_set:
            pic_path = pics_path + '/' + pic
            img = io.imread(pic_path).astype(np.double)
            img = color.rgb2gray(img)
            img = transform.resize(img, (32, 32), mode='constant')
            test_x.append(img.reshape(-1))
            lab = pic.split('.')[0]
            lab = lab.split('_')[1]
            lab = lab[0:2]
            label = int(lab)
            if label == 4:
                label = [1, 0, 0, 0, 0]
            elif label == 9:
                label = [0, 1, 0, 0, 0]
            elif label == 11:
                label = [0, 0, 1, 0, 0]
            elif label == 14:
                label = [0, 0, 0, 1, 0]
            elif label == 16:
                label = [0, 0, 0, 0, 1]
            test_y.append(label)

    train_x = np.asarray(train_x)
    train_y = np.asarray(train_y)
    test_x = np.asarray(test_x)
    test_y = np.asarray(test_y)

    train_x = train_x / 255
    test_x = test_x / 255
    return train_x, train_y, test_x, test_y


if __name__ == "__main__":
    source_train_x, source_train_y, source_test_x, source_test_y = import_data(100)
    target_train_x, target_train_y, target_test_x, target_test_y = import_data(100)
