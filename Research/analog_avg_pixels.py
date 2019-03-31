from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
import config


class TotalPixels(object):
    total_colors = []
    total_list = []
    total_list_l = []
    total_list_h = []
    total_colors_h = []
    total_colors_l = []


class ImageCollectionInfo(object):
    def __init__(self, name, mean):
        self.name = name
        self.mean = mean
        self.imgs = []

        self.avg_colors = []
        self.total_colors = []

        self.avg_colors_lower = []
        self.total_colors_lower = []

        self.avg_colors_higher = []
        self.total_colors_higher = []

    def add(self, img):
        self.imgs.append(img)

        self.avg_colors.append(img.mean)
        TotalPixels.total_colors.append(img.mean)

        self.total_colors += img.pixels
        TotalPixels.total_list += img.pixels

        if img.mean < self.mean:
            self.avg_colors_lower.append(img.mean)
            TotalPixels.total_colors_l.append(img.mean)

            self.total_colors_lower += img.pixels
            TotalPixels.total_list_l += img.pixels

        else:
            self.avg_colors_higher.append(img.mean)
            TotalPixels.total_colors_h.append(img.mean)

            self.total_colors_higher += img.pixels
            TotalPixels.total_list_h += img.pixels

    def show_graph(self, colors):
        data_list = {
            '_avg': {
                'color': colors[0],
                'bins': 50,
                'x': self.avg_colors
            },
            '_avg_l': {
                'color': colors[1],
                'bins': 25,
                'x': self.avg_colors_lower
            },
            '_avg_h': {
                'color': colors[2],
                'bins': 25,
                'x': self.avg_colors_higher
            },
            '_pixels': {
                'color': colors[3],
                'bins': 50,
                'x': self.total_colors
            },
            '_pixels_l': {
                'color': colors[4],
                'bins': 50,
                'x': self.total_colors_lower
            },
            '_pixels_h': {
                'color': colors[5],
                'bins': 50,
                'x': self.total_colors_higher
            },
        }

        for name, value in data_list.items():
            title = self.name + name

            plt.title(title)
            plt.hist(**value)
            plt.savefig('{}/{}.png'.format('plots', title))
            plt.show()

    def show_add_info(self):
        train_mean = np.mean(self.avg_colors)
        train_mean_l = np.mean(self.avg_colors_lower)
        train_mean_h = np.mean(self.avg_colors_higher)

        print('{} mean:'.format(self.name), train_mean)
        print('{} mean_l:'.format(self.name), train_mean_l)
        print('{} mean_h:'.format(self.name), train_mean_h)
        print('{} mean_lh:'.format(self.name), (train_mean_l + train_mean_h) / 2)
        print('{}_l:'.format(self.name), len(self.avg_colors_lower), 'train_h:', len(self.avg_colors_higher))

    def show_info(self, colors):

        self.show_graph(colors)

        self.show_add_info()


class ImageInfo(object):
    def __init__(self, dest):
        self.dest = dest
        self.pixels = []
        self.mean = None

        self._process_image()

    def _process_image(self):
        im = Image.open(self.dest)
        im_rgb = im.convert('RGB')

        for i in range(50):
            for j in range(50):
                red, _, _ = im_rgb.getpixel((i, j))
                self.pixels += [red]

        self.mean = np.mean(self.pixels)


dests_dict = {
    'train': {
        'dests': ['{}/{}'.format(config.TO_TRAIN_DATA, name) for name in os.listdir(config.TO_TRAIN_DATA)],
        'mean': 69.15410413030831,
        'colors': ['red'] * 3 + ['orange'] * 3
    },
    'value': {
        'dests': ['{}/{}'.format(config.TO_VALUE_DATA, name) for name in os.listdir(config.TO_VALUE_DATA)],
        'mean': 67.3797933693138,
        'colors': ['cyan'] * 3 + ['pink'] * 3
    },
    'test': {
        'dests': ['{}/{}'.format(config.TO_TEST_DATA, name) for name in os.listdir(config.TO_TEST_DATA)],
        'mean': 70.04921331689272,
        'colors': ['blue'] * 3 + ['purple'] * 3
    },

}
collection_list = []
for name, obj in dests_dict.items():
    img_collection = ImageCollectionInfo(name, obj['mean'])

    for dest in obj['dests']:
        img = ImageInfo(dest)
        img_collection.add(img)

    img_collection.show_info(obj['colors'])
    collection_list.append(img_collection)

plt.title('total_avg')
plt.hist(TotalPixels.total_colors, bins=50, color='cyan')
plt.savefig('plots/total_avg.png')
plt.show()
plt.title('total_avg_l')
plt.hist(TotalPixels.total_colors_l, bins=25, color='cyan')
plt.savefig('plots/total_avg_l.png')
plt.show()
plt.title('total_avg_h')
plt.hist(TotalPixels.total_colors_h, bins=25, color='cyan')
plt.savefig('plots/total_avg_h.png')
plt.show()
plt.title('total_pixels')
plt.hist(TotalPixels.total_list, bins=50, color='pink')
plt.savefig('plots/total_pixels.png')
plt.show()
plt.title('total_pixels_l')
plt.hist(TotalPixels.total_list_l, bins=50, color='pink')
plt.savefig('plots/total_pixels_l.png')
plt.show()
plt.title('total_pixels_h')
plt.hist(TotalPixels.total_list_h, bins=50, color='pink')
plt.savefig('plots/total_pixels_h.png')
plt.show()
print('total mean:', np.mean(TotalPixels.total_colors))
print('total mean_l:', np.mean(TotalPixels.total_colors_l))
print('total mean_h:', np.mean(TotalPixels.total_colors_h))
print('total mean_lh:', (np.mean(TotalPixels.total_colors_l) + np.mean(TotalPixels.total_colors_h)) / 2)
print('total_l:', len(collection_list[0].total_colors + collection_list[1].total_colors + collection_list[2].total_colors),
      'total_h', len(collection_list[0].total_colors_higher + collection_list[1].total_colors_higher + collection_list[2].total_colors_higher))
