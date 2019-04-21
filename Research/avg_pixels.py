from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import config

train_labels = pd.read_csv('../CancerDetection/data/traindatalabels.txt', delimiter=' ', header=None)
train_labels.columns = ['img', 'target']
train_labels['mean'] = 0
value_labels = pd.read_csv('../CancerDetection/data/valdatalabels.txt', delimiter=' ', header=None)
value_labels.columns = ['img', 'target']
value_labels['mean'] = 0
test_labels = pd.read_csv('../CancerDetection/data/testdatalabels.txt', delimiter=' ', header=None)
test_labels.columns = ['img', 'target']
test_labels['mean'] = 0

train_colors, test_colors, value_colors, total_colors = [], [], [], []
train_list, test_list, value_list, total_list = [], [], [], []
train_colors_l, test_colors_l, value_colors_l, total_colors_l = [], [], [], []
train_list_l, test_list_l, value_list_l, total_list_l = [], [], [], []
train_colors_h, test_colors_h, value_colors_h, total_colors_h = [], [], [], []
train_list_h, test_list_h, value_list_h, total_list_h = [], [], [], []
train_targetsum_h, train_targetsum_l, value_targetsum_h, value_targetsum_l, test_targetsum_h, test_targetsum_l = 0, 0, 0, 0, 0, 0

train_dests = [config.TO_TRAIN_DATA + "/" + i for i in os.listdir(config.TO_TRAIN_DATA)]

for dest in train_dests:
    img_name = dest[dest.find('/') + 1:]
    img_target = train_labels[train_labels['img'] == img_name]['target'].sum()
    im = Image.open(dest)
    im_rgb = im.convert('RGB')
    temp_list = []
    for i in range(50):
        for j in range(50):
            r, g, b = im_rgb.getpixel((i, j))
            temp_list += [r]
    # plt.hist(temp_list, bins=50, color='black')
    # plt.show()
    train_labels['mean'] = np.where(train_labels['img'] == img_name, np.mean(temp_list), train_labels['mean'])
    train_colors += [np.mean(temp_list)]
    total_colors += [np.mean(temp_list)]
    total_list += temp_list
    train_list += temp_list
    if (np.mean(temp_list) < 69.15410413030831):
        train_targetsum_l += img_target
        train_colors_l += [np.mean(temp_list)]
        total_colors_l += [np.mean(temp_list)]
        total_list_l += temp_list
        train_list_l += temp_list
    else:
        train_targetsum_h += img_target
        train_colors_h += [np.mean(temp_list)]
        total_colors_h += [np.mean(temp_list)]
        total_list_h += temp_list
        train_list_h += temp_list
train_mean = np.mean(train_colors)
train_mean_l = np.mean(train_colors_l)
train_mean_h = np.mean(train_colors_h)

hist, bin_edges = np.histogram(train_labels['mean'], bins=20, density=True)
train_labels['bin'] = [np.searchsorted(bin_edges, ele) - 1 for ele in train_labels['mean']]
plt.title('train target_ratio on brightness')
plt.plot(bin_edges[1:], train_labels.groupby(by=['bin'])['target'].mean())
plt.savefig('plots/train_target_ratio.png')
plt.show()

plt.title('train_avg')
plt.hist(train_colors, bins=50, color='red')
plt.savefig('plots/train_avg.png')
plt.show()
# plt.title('train_avg_l')
# plt.hist(train_colors_l, bins=25, color='red')
# plt.savefig('plots/train_avg_l.png')
# plt.show()
# plt.title('train_avg_h')
# plt.hist(train_colors_h, bins=25, color='red')
# plt.savefig('plots/train_avg_h.png')
# plt.show()
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

print('train med:', np.median(train_colors))
print('train mean:', train_mean)
print('train mean_l:', train_mean_l)
print('train mean_h:', train_mean_h)
print('train mean_lh:', (train_mean_l + train_mean_h) / 2)
print('train_l:', len(train_colors_l), 'train_h:', len(train_colors_h))
print('train_ratio_l:', train_targetsum_l / len(train_colors_l), 'train_ratio_h:',
      train_targetsum_h / len(train_colors_h))

val_dests = [config.TO_VALUE_DATA + "/" + i for i in os.listdir(config.TO_VALUE_DATA)]

for dest in val_dests:
    img_name = dest[dest.find('/') + 1:]
    img_target = value_labels[value_labels['img'] == img_name]['target'].sum()
    im = Image.open(dest)
    im_rgb = im.convert('RGB')
    temp_list = []
    for i in range(50):
        for j in range(50):
            r, g, b = im_rgb.getpixel((i, j))
            temp_list += [r]
    value_labels['mean'] = np.where(value_labels['img'] == img_name, np.mean(temp_list), value_labels['mean'])
    value_colors += [np.mean(temp_list)]
    total_colors += [np.mean(temp_list)]
    total_list += temp_list
    value_list += temp_list
    if (np.mean(temp_list) < 67.3791933693138):
        value_targetsum_l += img_target
        value_colors_l += [np.mean(temp_list)]
        total_colors_l += [np.mean(temp_list)]
        total_list_l += temp_list
        value_list_l += temp_list
    else:
        value_targetsum_h += img_target
        value_colors_h += [np.mean(temp_list)]
        total_colors_h += [np.mean(temp_list)]
        total_list_h += temp_list
        value_list_h += temp_list
value_mean = np.mean(value_colors)
value_mean_l = np.mean(value_colors_l)
value_mean_h = np.mean(value_colors_h)

hist, bin_edges = np.histogram(value_labels['mean'], bins=20, density=True)
value_labels['bin'] = [np.searchsorted(bin_edges, ele) - 1 for ele in value_labels['mean']]
plt.title('value target_ratio on brightness')
plt.plot(bin_edges[1:], value_labels.groupby(by=['bin'])['target'].mean())
plt.savefig('plots/value_target_ratio.png')
plt.show()

plt.title('value_avg')
plt.hist(value_colors, bins=50, color='yellow')
plt.savefig('plots/value_avg.png')
plt.show()
# plt.title('value_avg_l')
# plt.hist(value_colors_l, bins=25, color='yellow')
# plt.savefig('plots/value_avg_l.png')
# plt.show()
# plt.title('value_avg_h')
# plt.hist(value_colors_h, bins=25, color='yellow')
# plt.savefig('plots/value_avg_h.png')
# plt.show()
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

print('value med:', np.median(value_colors))
print('value mean:', value_mean)
print('value mean_l:', value_mean_l)
print('value mean_h:', value_mean_h)
print('value mean_lh:', (value_mean_l + value_mean_h) / 2)
print('value_l:', len(value_colors_l), 'value_h:', len(value_colors_h))
print('value_ratio_l:', value_targetsum_l / len(value_colors_l), 'value_ratio_h:',
      value_targetsum_h / len(value_colors_h))

test_dests = [config.TO_TEST_DATA + "/" + i for i in os.listdir(config.TO_TEST_DATA)]

for dest in test_dests:
    img_name = dest[dest.find('/') + 1:]
    img_target = test_labels[test_labels['img'] == img_name]['target'].sum()
    im = Image.open(dest)
    im_rgb = im.convert('RGB')
    temp_list = []
    for i in range(50):
        for j in range(50):
            r, g, b = im_rgb.getpixel((i, j))
            temp_list += [r]
    test_labels['mean'] = np.where(test_labels['img'] == img_name, np.mean(temp_list), test_labels['mean'])
    test_colors += [np.mean(temp_list)]
    total_colors += [np.mean(temp_list)]
    total_list += temp_list
    test_list += temp_list
    if (np.mean(temp_list) < 70.04921331689272):
        test_targetsum_l += img_target
        test_colors_l += [np.mean(temp_list)]
        total_colors_l += [np.mean(temp_list)]
        total_list_l += temp_list
        test_list_l += temp_list
    else:
        test_targetsum_h += img_target
        test_colors_h += [np.mean(temp_list)]
        total_colors_h += [np.mean(temp_list)]
        total_list_h += temp_list
        test_list_h += temp_list
test_mean = np.mean(test_colors)
test_mean_l = np.mean(test_colors_l)
test_mean_h = np.mean(test_colors_h)

hist, bin_edges = np.histogram(test_labels['mean'], bins=20, density=True)
test_labels['bin'] = [np.searchsorted(bin_edges, ele) - 1 for ele in test_labels['mean']]
plt.title('test target_ratio on brightness')
plt.plot(bin_edges, test_labels.groupby(by=['bin'])['target'].mean())
plt.savefig('plots/test_target_ratio.png')
plt.show()

plt.title('test_avg')
plt.hist(test_colors, bins=50, color='blue')
plt.savefig('plots/test_avg.png')
plt.show()
# plt.title('test_avg_l')
# plt.hist(test_colors_l, bins=25, color='blue')
# plt.savefig('plots/test_avg_l.png')
# plt.show()
# plt.title('test_avg_h')
# plt.hist(test_colors_h, bins=25, color='blue')
# plt.savefig('plots/test_avg_h.png')
# plt.show()
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

print('test med:', np.median(test_colors))
print('test mean:', test_mean)
print('test mean_l:', test_mean_l)
print('test mean_h:', test_mean_h)
print('test mean_lh:', (test_mean_l + test_mean_h) / 2)
print('test_l:', len(test_colors_l), 'test_h:', len(test_colors_h))
print('test_ratio_l:', test_targetsum_l / len(test_colors_l), 'test_ratio_h:',
      test_targetsum_h / len(test_colors_h))

total_labels = pd.concat([train_labels, value_labels, test_labels])
hist, bin_edges = np.histogram(total_labels['mean'], bins=20, density=True)
total_labels['bin'] = [np.searchsorted(bin_edges, ele) - 1 for ele in total_labels['mean']]
plt.title('total target_ratio on brightness')
plt.plot(bin_edges[1:], total_labels.groupby(by=['bin'])['target'].mean())
plt.savefig('plots/total_target_ratio.png')
plt.show()

plt.title('total_avg')
plt.hist(total_colors, bins=50, color='cyan')
plt.savefig('plots/total_avg.png')
plt.show()
# plt.title('total_avg_l')
# plt.hist(total_colors_l, bins=25, color='cyan')
# plt.savefig('plots/total_avg_l.png')
# plt.show()
# plt.title('total_avg_h')
# plt.hist(total_colors_h, bins=25, color='cyan')
# plt.savefig('plots/total_avg_h.png')
# plt.show()
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

print('total med:', np.median(total_colors))
print('total mean:', np.mean(total_colors))
print('total mean_l:', np.mean(total_colors_l))
print('total mean_h:', np.mean(total_colors_h))
print('total mean_lh:', (np.mean(total_colors_l) + np.mean(total_colors_h)) / 2)
print('total_l:', len(train_colors_l + value_colors_l + test_colors_l), 'total_h:',
      len(train_colors_h + value_colors_h + test_colors_h))

plt.hist([total_list_l, total_list_h], bins=50, color=['red', 'blue'])
plt.savefig('plots/compare.png')
plt.show()

total_low = []
total_high = []
for pix in (total_list):
    if pix < 128:
        total_low += [pix]
    else:
        total_high += [pix]

print('mean:', np.mean(total_list))
print('std:', np.std(total_list))
print('mean <128:', np.mean(total_low))
print('std <128:', np.std(total_low))
print('mean >128:', np.mean(total_high))
print('std >128:', np.std(total_high))

rate_l = []
rate_h = []
for bright in range(25, 125):
    tmp_l = 0
    tmp_h = 0
    for dest in train_dests + val_dests + test_dests:
        im = Image.open(dest)
        im_rgb = im.convert('RGB')
        temp_list = []
        for i in range(50):
            for j in range(50):
                r, g, b = im_rgb.getpixel((i, j))
                temp_list += [r]
        pic_bright = np.mean(temp_list)
        if (pic_bright >= bright and pic_bright < bright + 1):
            for pix in temp_list:
                if pix < 128:
                    tmp_l += 1
                else:
                    tmp_h += 1
    rate_l += [tmp_l / (tmp_l + tmp_h)]
    rate_h += [tmp_h / (tmp_l + tmp_h)]

plt.plot([i for i in range(25, 125)], rate_l)
plt.plot([i for i in range(25, 125)], rate_h)
plt.savefig('plots/parts_ratio.png')
plt.show()
