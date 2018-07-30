# coding=utf-8

import os
import random
import time
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import dataset

bottleneck_tensor_size = 2048

bottleneck_tensor_name = 'pool_3/_reshape:0'

jpeg_data_tensor_name = 'DecodeJpeg/contents:0'

model_dir = './model'

model_file = 'tensorflow_inception_graph.pb'

train_input_data = './data/train'

save_model_dir = './save_model'

learning_rate = 0.01
steps = 200
batch = 500


def get_random_cached_bottleneck(sess, n_class, image_list, how_many, category, image_label, image_tensor,
                                 bottleneck_tensor):
    bottlenecks = []
    ground_truths = []

    for _ in range(how_many):
        image_index = random.randrange(65536)
        image_path, bottleneck = dataset.get_or_create_bottleneck(sess, image_list, train_input_data, image_index,
                                                                  category, image_tensor, bottleneck_tensor)

        ground_truth = np.zeros(n_class, dtype=np.float32)
        image_name = os.path.basename(image_path)
        label_index = image_label[image_name.split('.')[0]]
        ground_truth[label_index] = 1.0
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)

    return bottlenecks, ground_truths


def main(_):
    print('--------- create image list ---------')
    t = time.time()
    image_list = dataset.create_image_list(train_input_data)
    print('time:{:.2f} s'.format(time.time() - t))

    print('--------- create labels ----------')
    t = time.time()
    image_label, labels_set = dataset.create_label()
    n_class = len(labels_set)
    print('time:{:.2f} s'.format(time.time() - t))

    print('--------- load inception-v3 model -----------')
    t = time.time()
    with gfile.FastGFile(os.path.join(model_dir, model_file), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    bottleneck_tensor, image_tensor = tf.import_graph_def(graph_def, return_elements=[bottleneck_tensor_name,
                                                                                      jpeg_data_tensor_name])
    print('bottleneck_tensor shape:{}'.format(bottleneck_tensor.shape))
    print(bottleneck_tensor)
    print('image tensor shape:{}'.format(image_tensor.shape))
    print(image_tensor)

    bottleneck_input = tf.placeholder(tf.float32, [None, bottleneck_tensor_size], name='bottleneckInputPlaceholder')
    ground_truth_input = tf.placeholder(tf.float32, [None, n_class], name='groundTruthInputPlaceholder')

    with tf.name_scope('final_training_ops'):
        weights = tf.Variable(tf.truncated_normal([bottleneck_tensor_size, n_class], stddev=0.001))
        biases = tf.Variable(tf.zeros([n_class]))
        logits = tf.matmul(bottleneck_input, weights) + biases
        final_tensor = tf.nn.softmax(logits)
        tf.add_to_collection('predict_pro', final_tensor)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=ground_truth_input)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy_mean)

    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(final_tensor, 1), tf.argmax(ground_truth_input, 1))
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print('time:{:.2f} s'.format(time.time() - t))

    saver = tf.train.Saver()

    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)

        print('--------- training ----------')
        start = time.time()
        best_acc = 0
        last_time = start
        for i in range(steps):
            train_bottleneck, train_ground_truth = get_random_cached_bottleneck(sess, n_class, image_list, batch,
                                                                                'train', image_label, image_tensor,
                                                                                bottleneck_tensor)
            sess.run(train_step, feed_dict={bottleneck_input: train_bottleneck, ground_truth_input: train_ground_truth})

            if i % 100 == 0 or i == steps - 1:
                validation_bottleneck, validation_ground_truth = get_random_cached_bottleneck(sess, n_class, image_list,
                                                                                              batch,
                                                                                              'valid', image_label,
                                                                                              image_tensor,
                                                                                              bottleneck_tensor)
                valid_accuracy = sess.run(evaluation_step, feed_dict={bottleneck_input: validation_bottleneck,
                                                                      ground_truth_input: validation_ground_truth})

                print('Step %-4d: Validation accuracy on random sampled %4d examples = %.1f%%' % (
                    i, batch, valid_accuracy * 100))
                if valid_accuracy > best_acc:
                    best_acc = valid_accuracy
                    saver.save(sess, os.path.join(save_model_dir, 'model-{:.2f}.ckpt'.format(valid_accuracy)))

                print('time:{:.2f} s'.format(time.time() - last_time))
                last_time = time.time()

        print('total train time:{:.2f} s'.format(time.time() - start))


if __name__ == '__main__':
    tf.app.run()
