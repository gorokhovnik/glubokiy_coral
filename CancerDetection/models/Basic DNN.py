from PIL import Image
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
import config
import numpy as np
import pandas as pd
import tensorflow as tf
import tflearn
import random

train_labels = pd.read_csv('../data/traindatalabels.txt', delimiter=' ', header=None)
train_labels.columns = ['img', 'target']
value_labels = pd.read_csv('../data/valdatalabels.txt', delimiter=' ', header=None)
value_labels.columns = ['img', 'target']
test_labels = pd.read_csv('../data/testdatalabels.txt', delimiter=' ', header=None)
test_labels.columns = ['img', 'target']

mean_total = 69.06738499234304
std_total = 68.45855223205473
mean_low = 34.77274047733289
std_low = 24.00922853254174
mean_high = 187.0906383787584
std_high = 29.22877675953323

def create_feature_sets_and_labels(data='train'):
    x = []
    y = []
    if data == 'train':
        dests = ['../data/train/1' + "/" + i for i in os.listdir('../data/train/1')]
    elif data == 'val':
        dests = ['../data/val/1' + "/" + i for i in os.listdir('../data/val/1')]
    elif data == 'test':
        dests = ['../data/test/1' + "/" + i for i in os.listdir('../data/test/1')]

    for id, dest in enumerate(dests):
        img_name = dest[dest.rfind('/') + 1:]
        im = Image.open(dest)
        im_rgb = im.convert('RGB')
        tmp_l = []
        tmp_h = []
        pic_list = []
        for i in range(50):
            temp_list = []
            for j in range(50):
                r, g, b = im_rgb.getpixel((i, j))
                temp_list += [r]
                if r < 128:
                    tmp_l += [r]
                else:
                    tmp_h += [r]
            pic_list += [temp_list]
        low_mean = np.mean(tmp_l)
        low_std = np.std(tmp_l)
        high_mean = np.mean(tmp_h)
        high_std = np.std(tmp_h)
        total_mean = np.mean(tmp_l + tmp_h)
        total_std = np.std(tmp_l + tmp_h)
        if low_std == 0:
            low_std = std_low
        if high_std == 0:
            high_std = std_high
        if total_std == 0:
            total_std = std_total

        for i in range(50):
            for j in range(50):
                # pass
                # pic_list[i][j] = (pic_list[i][j] - total_mean) / total_std

                if pic_list[i][j] < 128:
                    pic_list[i][j] = (pic_list[i][j] - low_mean) / low_std - 3
                else:
                    pic_list[i][j] = (pic_list[i][j] - high_mean) / high_std + 3

        if data == 'train':
            target = train_labels[train_labels['img'] == img_name]['target'].sum()
        elif data == 'val':
            target = value_labels[value_labels['img'] == img_name]['target'].sum()
        elif data == 'test':
            target = test_labels[test_labels['img'] == img_name]['target'].sum()

        x += [pic_list]
        y += [[target, 1 - target]]
    return x, y


train_x, train_y = create_feature_sets_and_labels('train')
value_x, value_y = create_feature_sets_and_labels('val')
test_x, test_y = create_feature_sets_and_labels('test')
tmp_x, tmp_y = train_x[:3500] + value_x[:600] + test_x[:900], train_y[:3500] + value_y[:600] + test_y[:900]
value_x = train_x[3500:] + value_x[600:] + test_x[900:]
value_y = train_y[3500:] + value_y[600:] + test_y[900:]
train_x = tmp_x
train_y = tmp_y

train_AUC = []
value_AUC = []

tf.reset_default_graph()
tflearn.init_graph(seed=228)

net = tflearn.input_data(shape=[None, 50, 50])
net = tflearn.fully_connected(net, 32, name='hidden1')
net = tflearn.fully_connected(net, 32, name='hidden2')
net = tflearn.fully_connected(net, 2, activation='softmax', name='output')
net = tflearn.regression(net, shuffle_batches=False)

model = tflearn.DNN(net, tensorboard_verbose=None)

for ep in range(10):
    model.fit(train_x, train_y, validation_set=(value_x, value_y), n_epoch=1, batch_size=20)

    train_p = np.array(model.predict(train_x))[:, 0]
    value_p = np.array(model.predict(value_x))[:, 0]
    train_AUC += [roc_auc_score(np.array(train_y)[:, 0], train_p)]
    value_AUC += [roc_auc_score(np.array(value_y)[:, 0], value_p)]

AUCs = pd.DataFrame()
AUCs['train'] = pd.Series(train_AUC)
AUCs['value'] = pd.Series(value_AUC)
print(AUCs)