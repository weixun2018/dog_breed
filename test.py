# coding=utf-8

import os
import pandas as pd
import tensorflow as tf
import time
import dataset

bottleneck_tensor_size = 2048

submit_file = 'submission.csv'

test_input_data = './data/test'

bottleneck_tensor_name = 'import/pool_3/_reshape:0'

jpeg_data_tensor_name = 'import/DecodeJpeg/contents:0'

save_model_dir = './save_model'

save_model = 'model.pb'


def get_test_bottlenecks(sess, image_list, image_tensor, bottleneck_tensor):
    bottlenecks = []
    image_path_list = []

    category = 'test'
    for index, _ in enumerate(image_list[category]):
        image_path, bottleneck = dataset.get_or_create_bottleneck(sess, image_list, test_input_data, index, category,
                                                                  image_tensor,
                                                                  bottleneck_tensor)
        image_path_list.append(image_path)
        bottlenecks.append(bottleneck)

    return image_path_list, bottlenecks


def submit(images, predict, labels_set):
    test_result = []
    for img, output_tensor in zip(images, predict):
        img_id = os.path.basename(img).split('.')[0]
        output_str = [str(x) for x in output_tensor]
        test_result.append([img_id] + output_str)

    header = ['id']
    header.extend(labels_set)
    print('header:', header)
    test_result_df = pd.DataFrame(test_result)
    print('test result df shape:', test_result_df.shape)
    with open(submit_file, 'w') as sub_file:
        test_result_df.to_csv(sub_file, sep=',', header=header, index=False)


def main(_):
    print('--------- create test image list ---------')
    t = time.time()
    test_image_list = dataset.create_image_list(test_input_data, flag='test')
    print('time:{:.2f} s'.format(time.time() - t))

    print('--------- create labels ----------')
    t = time.time()
    image_label, labels_set = dataset.create_label()
    print('time:{:.2f} s'.format(time.time() - t))

    print('--------- test ------')
    ckpt = tf.train.latest_checkpoint(save_model_dir)
    print('model path: ', ckpt)

    saver = tf.train.import_meta_graph(ckpt + '.meta')

    with tf.Session() as sess:
        saver.restore(sess, ckpt)
        final_tensor = tf.get_collection('predict_pro')[0]
        print(final_tensor)

        graph = tf.get_default_graph()

        # for ops in graph.get_operations():
        #     print(ops.name, ops.values())

        image_tensor = graph.get_tensor_by_name(jpeg_data_tensor_name)
        bottleneck_tensor = graph.get_tensor_by_name(bottleneck_tensor_name)
        bottleneck_input = graph.get_operation_by_name('bottleneckInputPlaceholder').outputs[0]
        print(graph.get_operation_by_name('bottleneckInputPlaceholder').outputs)
        print(bottleneck_input)

        print('--------- test ----------')
        t = time.time()
        test_images, test_bottleneck = get_test_bottlenecks(sess, test_image_list, image_tensor, bottleneck_tensor)
        print('test_bottleneck shape:', len(test_bottleneck), len(test_bottleneck[0]))
        print('get test images bottleneck tensor, quantity:{}, time:{:.2f} s'.format(len(test_images), time.time() - t))
        predict_pro = sess.run(final_tensor, feed_dict={bottleneck_input: test_bottleneck})
        print('finish prediction, time:{:.2f} s'.format(time.time() - t))
        print('predict pro shape:', predict_pro.shape)
        submit(test_images, predict_pro, labels_set)
        print('time:{:.2f} s'.format(time.time() - t))


if __name__ == '__main__':
    tf.app.run()
