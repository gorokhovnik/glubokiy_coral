from PIL import Image
import os
import matplotlib.pyplot as plt

import config

train_colors, test_colors, value_test = [], [], []

temp_list = []
test_dests = [config.TO_TRAIN_DATA + "/" + i for i in os.listdir(config.TO_TRAIN_DATA)]
for dest in test_dests:
    im = Image.open(dest)
    im_rgb = im.convert('RGB')
    for i in range(50):
        for j in range(50):
            r, g, b = im_rgb.getpixel((i, j))
            temp_list.append([r, g, b])

    train_colors += [sum([i[0] for i in temp_list]) / len([i[0] for i in temp_list])]

temp_list = []
test_dests = [config.TO_TEST_DATA + "/" + i for i in os.listdir(config.TO_TEST_DATA)]
for dest in test_dests:
    im = Image.open(dest)
    im_rgb = im.convert('RGB')

    for i in range(50):
        for j in range(50):
            r, g, b = im_rgb.getpixel((i, j))
            temp_list.append([r, g, b])

    test_colors += [sum([i[0] for i in temp_list]) / len([i[0] for i in temp_list])]
#
val_dests = [config.TO_VALUE_DATA + "/" + i for i in os.listdir(config.TO_VALUE_DATA)]
temp_list = []
for dest in val_dests:
    im = Image.open(dest)
    im_rgb = im.convert('RGB')

    for i in range(50):
        for j in range(50):
            r, g, b = im_rgb.getpixel((i, j))
            temp_list.append([r, g, b])

    value_test += [sum([i[0] for i in temp_list]) / len([i[0] for i in temp_list])]
# plt.hist(train_clors)
# plt.hist(test_colors)
colors = ['Red', 'Yellow', 'Green']
names = ['value', 'test', 'train']
for nonuse, j in enumerate([value_test]):
    plt.title(names[nonuse])

    plt.hist([int(i) for i in j], bins=50, color=colors[nonuse])
    plt.show()
    # plt.hist([i[1] for i in j], bins=50)
    # plt.show()
    # plt.hist([i[2] for i in j], bins=50)
    # plt.show()

plt.title('Total')
plt.hist([int(i) for i in value_test + test_colors + train_colors], bins=50, color='Purple')
plt.show()
# plt.hist([i[1] for i in value_test+test_colors+train_clors], bins=50)
# plt.show()
# plt.hist([i[2] for i in value_test+test_colors+train_clors], bins=50)
# plt.show()


names = ['train', 'value', 'test']
for i, j in enumerate([train_colors, value_test, test_colors]):
    print("Start avg %s" % names[i])
    print(sum([i[0] for i in j]) / len([i[0] for i in j]))
    print(sum([i[1] for i in j]) / len([i[1] for i in j]))
    print(sum([i[2] for i in j]) / len([i[2] for i in j]))
    print("End \n\n")

print("All avgs")
print(sum([i[0] for i in value_test + test_colors + train_colors]) / len(
    [i[0] for i in value_test + test_colors + train_colors]))
print(sum([i[1] for i in value_test + test_colors + train_colors]) / len(
    [i[1] for i in value_test + test_colors + train_colors]))
print(sum([i[2] for i in value_test + test_colors + train_colors]) / len(
    [i[2] for i in value_test + test_colors + train_colors]))