import os
from shutil import copyfile

import config
from Research.analog_avg_pixels import ImageInfo

dests_dict = {
    'train': {
        # List of absolute destination to pictures
        'dests': ['{}/{}'.format(config.TO_TRAIN_DATA, name) for name in os.listdir(config.TO_TRAIN_DATA)],
        # Mean filter
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

# Iter tor train, value and test data
for key, value in dests_dict.items():
    count = len(value['dests'])
    # Iter in every dest in train, value, test data
    for i, dest in enumerate(value['dests']):
        img = ImageInfo(dest)

        # Get name of file which will copy
        name = dest.split('/')[-1]
        # Get mean and choose the folder (greater or less)
        mean_type = 'higher' if img.mean < value['mean'] else 'lower'
        # Process new destination
        new_dest = '{}/{}/{}/{}'.format(dest_to_target, mean_type, key, name)
        # Copying
        copyfile(dest, new_dest)

        # Lof of work
        print('[{}/{}] {} {} -> {}'.format(
            i, count, mean_type.capitalize().ljust(6, ' '), dest.ljust(95, ' '), new_dest))



