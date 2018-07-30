# coding=utf-8

import os
import numpy as np
import pandas as pd
import glob
from tensorflow.python.platform import gfile

cache_dir = './data/cache'

train_input_data = './data/train'

label_file = './data/labels.csv'

test_input_data = './data/test'

validation_percentage = 10


def create_label():
    reader = pd.read_csv(label_file, sep=',', header=0)
    image_id = reader['id']
    dog_breed = reader['breed']
    labels_set = sorted(list(set(dog_breed)))
    print('the quantity of dog breed:{}'.format(len(labels_set)))
    # print(labels_set)
    image_label = {k: labels_set.index(v) for k, v in zip(image_id, dog_breed)}
    print('the quantity of dog image:{}'.format(len(image_label)))

    return image_label, labels_set


def one_hot(labels_set, image_name):
    arr = np.zeros(len(labels_set))
    arr[labels_set.index(image_name)] = 1

    return arr


def create_image_list(input_data, flag='train'):
    extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
    image_file_list = []
    for extension in extensions:
        image_file = os.path.join(input_data, '*.' + extension)
        image_file_list.extend(glob.glob(image_file))

    train_images = []
    valid_images = []
    test_images = []

    for file_name in image_file_list:
        base_name = os.path.basename(file_name)
        if flag == 'train':
            chance = np.random.randint(100)
            if chance < validation_percentage:
                valid_images.append(base_name)
            else:
                train_images.append(base_name)
        else:
            test_images.append(base_name)
    if flag == 'train':
        result = {'train': train_images, 'valid': valid_images}
        print('train images:{}\nvalid images:{}'.format(len(train_images), len(valid_images)))
    else:
        result = {'test': test_images}
        print('test images:{}'.format(len(test_images)))

    return result


def get_image_path(image_list, image_dir, index, category):
    category_list = image_list[category]
    mod_index = index % len(category_list)
    base_name = category_list[mod_index]

    full_path = os.path.join(image_dir, base_name)
    return full_path


def get_bottleneck_path(image_list, index, category):
    image_path = get_image_path(image_list, cache_dir, index, category)
    postfix = image_path.split('.')[-1]
    bottleneck_path = image_path.replace(str(postfix), 'txt')

    return bottleneck_path


def get_or_create_bottleneck(sess, image_list, image_dir, index, category, image_tensor, bottleneck_tensor):
    bottleneck_path = get_bottleneck_path(image_list, index, category)
    image_path = get_image_path(image_list, image_dir, index, category)
    if not os.path.exists(bottleneck_path):
        image = gfile.FastGFile(image_path, 'rb').read()
        bottleneck_value = run_bottleneck_on_image(sess, image, image_tensor, bottleneck_tensor)
        bottle_str = ','.join(str(x) for x in bottleneck_value)
        with open(bottleneck_path, 'w') as bottleneck_file:
            bottleneck_file.write(bottle_str)

    else:
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_str = bottleneck_file.read()
        bottleneck_value = [float(x) for x in bottleneck_str.split(',')]

    return image_path, bottleneck_value


def run_bottleneck_on_image(sess, image, image_tensor, bottleneck_tensor):
    bottleneck_value = sess.run(bottleneck_tensor, {image_tensor: image})
    bottleneck_value = np.squeeze(bottleneck_value)

    return bottleneck_value
