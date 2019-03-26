from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
import config

train_colors, test_colors, value_colors, total_colors = [], [], [], []
train_list, test_list, value_list, total_list = [], [], [], []
train_colors_l, test_colors_l, value_colors_l, total_colors_l = [], [], [], []
train_list_l, test_list_l, value_list_l, total_list_l = [], [], [], []
train_colors_h, test_colors_h, value_colors_h, total_colors_h = [], [], [], []
train_list_h, test_list_h, value_list_h, total_list_h = [], [], [], []

train_dests = [config.TO_TRAIN_DATA + "/" + i for i in os.listdir(config.TO_TRAIN_DATA)]

for dest in train_dests:
    im = Image.open(dest)
    im_rgb = im.convert('RGB')
    temp_list = []
    for i in range(50):
        for j in range(50):
            r, g, b = im_rgb.getpixel((i, j))
            temp_list += [r]
    train_colors += [np.mean(temp_list)]
    total_colors += [np.mean(temp_list)]
    total_list += temp_list
    train_list += temp_list
    if (np.mean(temp_list) < 69.15410413030831):
        train_colors_l += [np.mean(temp_list)]
        total_colors_l += [np.mean(temp_list)]
        total_list_l += temp_list
        train_list_l += temp_list
    else:
        train_colors_h += [np.mean(temp_list)]
        total_colors_h += [np.mean(temp_list)]
        total_list_h += temp_list
        train_list_h += temp_list
train_mean = np.mean(train_colors)
train_mean_l = np.mean(train_colors_l)
train_mean_h = np.mean(train_colors_h)

plt.title('train_avg')
plt.hist(train_colors, bins=50, color='red')
plt.savefig('plots/train_avg.png')
plt.show()
plt.title('train_avg_l')
plt.hist(train_colors_l, bins=25, color='red')
plt.savefig('plots/train_avg_l.png')
plt.show()
plt.title('train_avg_h')
plt.hist(train_colors_h, bins=25, color='red')
plt.savefig('plots/train_avg_h.png')
plt.show()
plt.title('train_pixels')
plt.hist(train_list, bins=50, color='orange')
plt.savefig('plots/train_pixels.png')
plt.show()
plt.title('train_pixels_l')
plt.hist(train_list_l, bins=50, color='orange')
plt.savefig('plots/train_pixels_l.png')
plt.show()
plt.title('train_pixels_h')
plt.hist(train_list_h, bins=50, color='orange')
plt.savefig('plots/train_pixels_h.png')
plt.show()
print('train mean:', train_mean)
print('train mean_l:', train_mean_l)
print('train mean_h:', train_mean_h)
print('train mean_lh:', (train_mean_l + train_mean_h) / 2)
print('train_l:', len(train_colors_l), 'train_h:', len(train_colors_h))

val_dests = [config.TO_VALUE_DATA + "/" + i for i in os.listdir(config.TO_VALUE_DATA)]

for dest in val_dests:
    im = Image.open(dest)
    im_rgb = im.convert('RGB')
    temp_list = []
    for i in range(50):
        for j in range(50):
            r, g, b = im_rgb.getpixel((i, j))
            temp_list += [r]
    value_colors += [np.mean(temp_list)]
    total_colors += [np.mean(temp_list)]
    total_list += temp_list
    value_list += temp_list
    if (np.mean(temp_list) < 67.3791933693138):
        value_colors_l += [np.mean(temp_list)]
        total_colors_l += [np.mean(temp_list)]
        total_list_l += temp_list
        value_list_l += temp_list
    else:
        value_colors_h += [np.mean(temp_list)]
        total_colors_h += [np.mean(temp_list)]
        total_list_h += temp_list
        value_list_h += temp_list
value_mean = np.mean(value_colors)
value_mean_l = np.mean(value_colors_l)
value_mean_h = np.mean(value_colors_h)

plt.title('value_avg')
plt.hist(value_colors, bins=50, color='yellow')
plt.savefig('plots/value_avg.png')
plt.show()
plt.title('value_avg_l')
plt.hist(value_colors_l, bins=25, color='yellow')
plt.savefig('plots/value_avg_l.png')
plt.show()
plt.title('value_avg_h')
plt.hist(value_colors_h, bins=25, color='yellow')
plt.savefig('plots/value_avg_h.png')
plt.show()
plt.title('value_pixels')
plt.hist(value_list, bins=50, color='green')
plt.savefig('plots/value_pixels.png')
plt.show()
plt.title('value_pixels_l')
plt.hist(value_list_l, bins=50, color='green')
plt.savefig('plots/value_pixels_l.png')
plt.show()
plt.title('value_pixels_h')
plt.hist(value_list_h, bins=50, color='green')
plt.savefig('plots/value_pixels_h.png')
plt.show()
print('value mean:', value_mean)
print('value mean_l:', value_mean_l)
print('value mean_h:', value_mean_h)
print('value mean_lh:', (value_mean_l + value_mean_h) / 2)
print('value_l:', len(value_colors_l), 'value_h', len(value_colors_h))

test_dests = [config.TO_TEST_DATA + "/" + i for i in os.listdir(config.TO_TEST_DATA)]

for dest in test_dests:
    im = Image.open(dest)
    im_rgb = im.convert('RGB')
    temp_list = []
    for i in range(50):
        for j in range(50):
            r, g, b = im_rgb.getpixel((i, j))
            temp_list += [r]
    test_colors += [np.mean(temp_list)]
    total_colors += [np.mean(temp_list)]
    total_list += temp_list
    test_list += temp_list
    if (np.mean(temp_list) < 70.04921331689272):
        test_colors_l += [np.mean(temp_list)]
        total_colors_l += [np.mean(temp_list)]
        total_list_l += temp_list
        test_list_l += temp_list
    else:
        test_colors_h += [np.mean(temp_list)]
        total_colors_h += [np.mean(temp_list)]
        total_list_h += temp_list
        test_list_h += temp_list
test_mean = np.mean(test_colors)
test_mean_l = np.mean(test_colors_l)
test_mean_h = np.mean(test_colors_h)

plt.title('test_avg')
plt.hist(test_colors, bins=50, color='blue')
plt.savefig('plots/test_avg.png')
plt.show()
plt.title('test_avg_l')
plt.hist(test_colors_l, bins=25, color='blue')
plt.savefig('plots/test_avg_l.png')
plt.show()
plt.title('test_avg_h')
plt.hist(test_colors_h, bins=25, color='blue')
plt.savefig('plots/test_avg_h.png')
plt.show()
plt.title('test_pixels')
plt.hist(test_list, bins=50, color='purple')
plt.savefig('plots/test_pixels.png')
plt.show()
plt.title('test_pixels_l')
plt.hist(test_list_l, bins=50, color='purple')
plt.savefig('plots/test_pixels_l.png')
plt.show()
plt.title('test_pixels_h')
plt.hist(test_list_h, bins=50, color='purple')
plt.savefig('plots/test_pixels_h.png')
plt.show()
print('test mean:', test_mean)
print('test mean_l:', test_mean_l)
print('test mean_h:', test_mean_h)
print('test mean_lh:', (test_mean_l + test_mean_h) / 2)
print('test_l:', len(test_colors_l), 'test_h', len(test_colors_h))

plt.title('total_avg')
plt.hist(total_colors, bins=50, color='cyan')
plt.savefig('plots/total_avg.png')
plt.show()
plt.title('total_avg_l')
plt.hist(total_colors_l, bins=25, color='cyan')
plt.savefig('plots/total_avg_l.png')
plt.show()
plt.title('total_avg_h')
plt.hist(total_colors_h, bins=25, color='cyan')
plt.savefig('plots/total_avg_h.png')
plt.show()
plt.title('total_pixels')
plt.hist(total_list, bins=50, color='pink')
plt.savefig('plots/total_pixels.png')
plt.show()
plt.title('total_pixels_l')
plt.hist(total_list_l, bins=50, color='pink')
plt.savefig('plots/total_pixels_l.png')
plt.show()
plt.title('total_pixels_h')
plt.hist(total_list_h, bins=50, color='pink')
plt.savefig('plots/total_pixels_h.png')
plt.show()
print('total mean:', np.mean(total_colors))
print('total mean_l:', np.mean(total_colors_l))
print('total mean_h:', np.mean(total_colors_h))
print('total mean_lh:', (np.mean(total_colors_l) + np.mean(total_colors_h)) / 2)
print('total_l:', len(train_colors_l + value_colors_l + test_colors_l), 'total_h', len(train_colors_h + value_colors_h + test_colors_h))