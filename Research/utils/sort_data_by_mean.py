import os
from shutil import copyfile

import config
from Research.analog_avg_pixels import ImageInfo

dests_dict = {
    'train': {
        'dests': ['{}/{}'.format(config.TO_TRAIN_DATA, name) for name in os.listdir(config.TO_TRAIN_DATA)],
        'mean': 69.15410413030831,
    },
    'value': {
        'dests': ['{}/{}'.format(config.TO_VALUE_DATA, name) for name in os.listdir(config.TO_VALUE_DATA)],
        'mean': 67.3797933693138,
    },
    'test': {
        'dests': ['{}/{}'.format(config.TO_TEST_DATA, name) for name in os.listdir(config.TO_TEST_DATA)],
        'mean': 70.04921331689272,
    },
}

dest_to_target = config.TO_ROOT + '/Research/data_prepared/sort_by_mean'

for key, value in dests_dict.items():
    count = len(value['dests'])
    for i, dest in enumerate(value['dests']):
        img = ImageInfo(dest)

        name = dest.split('/')[-1]
        mean_type = 'higher' if img.mean < value['mean'] else 'lower'

        new_dest = '{}/{}/{}/{}'.format(dest_to_target, mean_type, key, name)
        copyfile(dest, new_dest)

        print('[{}/{}] {} {} -> {}'.format(
            i, count, mean_type.capitalize().ljust(6, ' '), dest.ljust(95, ' '), new_dest))



