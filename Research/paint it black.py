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
        self.name = None

        self._process_image()

    def _get_mean(self):
        return np.mean(self.pixels)

    def _get_name(self):
        return self.dest.split('/')[-1]

    def _get_pixels(self):
        im = Image.open(self.dest)
        im_rgb = im.convert('RGB')
        pixels = []

        for i in range(50):
            for j in range(50):
                red, _, _ = im_rgb.getpixel((i, j))
                pixels += [red]

        return pixels

    def _process_image(self):
        self.name = self._get_name()
        self.pixels = self._get_pixels()
        self.mean = self._get_mean()


from PIL import Image
import config
import os
import time

from Research.analog_avg_pixels import ImageInfo


class ExtImageInfo(ImageInfo):
    def _get_mean(self):
        return None


def from_grayscale_list_to_img(data):
    img = Image.new(mode='L', size=(50, 50))

    for x in range(50):
        for y in range(50):
            img.putpixel((x, y), int(data[x][y]))

    return img


if __name__ == '__main__':
    mean_low = 35.266790886114705
    std_low = 23.81616202900302
    mean_high = 184.72147074982624
    std_high = 26.788517268439215
    dests_dict = {
        'train': {
            # List of absolute destination to pictures
            'dests': ['{}/{}'.format(config.TO_TRAIN_DATA, name) for name in os.listdir(config.TO_TRAIN_DATA)],
        },
        'value': {
            'dests': ['{}/{}'.format(config.TO_VALUE_DATA, name) for name in os.listdir(config.TO_VALUE_DATA)],
        },
        'test': {
            'dests': ['{}/{}'.format(config.TO_TEST_DATA, name) for name in os.listdir(config.TO_TEST_DATA)],
        },
    }
    t1 = time.time()
    for key, value in dests_dict.items():
        count = len(value['dests'])
        # Iter at every dest in train, value, test data
        for i, dest in enumerate(value['dests']):
            old_image = ExtImageInfo(dest)

            # GOROKHOV FUNCTION

            # Inverse data
            tmp_l = []
            tmp_h = []
            tmp_255 = []
            tmp_0 = []
            old_image = old_image.pixels
            for p in range(2500):
                if old_image[p] == 255:
                    tmp_255 += [old_image[p]]
                elif old_image[p] == 0:
                    tmp_0 += [old_image[p]]
                elif old_image[p] < 128:
                    tmp_l += [old_image[p]]
                else:
                    tmp_h += [old_image[p]]
            low_mean = np.mean(tmp_l)
            low_std = np.std(tmp_l)
            high_mean = np.mean(tmp_h)
            high_std = np.std(tmp_h)
            if low_std == 0:
                low_std = std_low
            if high_std == 0:
                high_std = std_high

            for p in range(2500):
                if old_image[p] != 0 and old_image[p] != 255:
                    if old_image[p] < 128:
                        old_image[p] = np.rint((old_image[p] - low_mean) / low_std * std_low + mean_low)
                    else:
                        old_image[p] = np.rint((old_image[p] - high_mean) / high_std * std_high + mean_high)


            # Convert from list [1..2500] to [[1..50] ... [1..50]]
            data = [old_image[50 * i: 50 * (i + 1)] for i in range(50)]

            # END GOROHOV

            new_image = from_grayscale_list_to_img(data)

            # new_dest = '{}/{}/{}'.format(dest_to_target, key, old_image.name)
            new_image.save(dest)

            print('[{}/{}] {}'.format(
                i, count, dest.ljust(95, ' ')))
    print(time.time() - t1)
