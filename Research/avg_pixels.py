from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
import config

train_colors, test_colors, value_colors, total_colors = [], [], [], []
train_list, test_list, value_list, total_list = [], [], [], []

train_dests = [config.TO_TRAIN_DATA + "/" + i for i in os.listdir(config.TO_TRAIN_DATA)]

for dest in train_dests:
    im = Image.open(dest)
    im_rgb = im.convert('RGB')
    temp_list = []
    for i in range(50):
        for j in range(50):
            r, g, b = im_rgb.getpixel((i, j))
            total_list += [r]
            train_list += [r]
            temp_list += [r]
    train_colors += [np.mean(temp_list)]
    total_colors += [np.mean(temp_list)]
train_mean = np.mean(train_colors)

plt.title('train_avg')
plt.hist(train_colors, bins=50, color='red')
plt.show()
plt.title('train_pixels')
plt.hist(train_list, bins=50, color='orange')
plt.show()
print('train mean: ', train_mean)

val_dests = [config.TO_VALUE_DATA + "/" + i for i in os.listdir(config.TO_VALUE_DATA)]

for dest in val_dests:
    im = Image.open(dest)
    im_rgb = im.convert('RGB')
    temp_list = []
    for i in range(50):
        for j in range(50):
            r, g, b = im_rgb.getpixel((i, j))
            total_list += [r]
            value_list += [r]
            temp_list += [r]
    value_colors += [np.mean(temp_list)]
    total_colors += [np.mean(temp_list)]
value_mean = np.mean(value_colors)

plt.title('value_avg')
plt.hist(value_colors, bins=50, color='yellow')
plt.show()
plt.title('value_pixels')
plt.hist(value_list, bins=50, color='green')
plt.show()
print('value mean: ', value_mean)

test_dests = [config.TO_TEST_DATA + "/" + i for i in os.listdir(config.TO_TEST_DATA)]

for dest in test_dests:
    im = Image.open(dest)
    im_rgb = im.convert('RGB')
    temp_list = []
    for i in range(50):
        for j in range(50):
            r, g, b = im_rgb.getpixel((i, j))
            total_list += [r]
            test_list += [r]
            temp_list += [r]
    test_colors += [np.mean(temp_list)]
    total_colors += [np.mean(temp_list)]
test_mean = np.mean(test_colors)

plt.title('test_avg')
plt.hist(test_colors, bins=50, color='blue')
plt.show()
plt.title('test_pixels')
plt.hist(test_list, bins=50, color='purple')
plt.show()
print('test mean: ', test_mean)

plt.title('total_avg')
plt.hist(total_colors, bins=50, color='cyan')
plt.show()
plt.title('total_pixels')
plt.hist(total_list, bins=50, color='pink')
plt.show()
print('total mean: ', np.mean(total_colors))
