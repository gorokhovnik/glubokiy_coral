from PIL import Image
import os
import matplotlib.pyplot as plt
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
        pic_list = []
        for i in range(50):
            temp_list = []
            for j in range(50):
                r, g, b = im_rgb.getpixel((i, j))
                temp_list += [r]
            pic_list += [temp_list]
        if data == 'train':
            target = train_labels[train_labels['img'] == img_name]['target'].sum()
        elif data == 'val':
            target = value_labels[value_labels['img'] == img_name]['target'].sum()
        elif data == 'test':
            target = test_labels[test_labels['img'] == img_name]['target'].sum()

        x += [pic_list]
        y += [[target, 1 - target]]
    return x, y


tf.reset_default_graph()
tflearn.init_graph(seed=228)

net = tflearn.input_data(shape=[None, 50, 50])

net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net, shuffle_batches=False)

train_x, train_y = create_feature_sets_and_labels('train')
value_x, value_y = create_feature_sets_and_labels('val')

model = tflearn.DNN(net, tensorboard_verbose=None)

model.fit(train_x, train_y, validation_set=(value_x, value_y), n_epoch=5, batch_size=16, show_metric=True)
