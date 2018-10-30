import tensorflow as tf
import numpy as np
from math import ceil
import os


def get_files(file_dir, test_size=0.2):
    """
    Args:
        file_dir: 文件夹路径
    Returns:
        乱序后的图片和标签的 list
    """
    cats = []
    label_cats = []
    dogs = []
    label_dogs = []
    for file in os.listdir(file_dir):
        name = file.split('.')
        if name[0] == 'cat':
            cats.append(file_dir + file)
            label_cats.append(0)
        else:
            dogs.append(file_dir + file)
            label_dogs.append(1)
    print('There are {} cats\nThere are {} dogs'.format(len(cats), len(dogs)))
    
    # shuffle
    image_list = np.hstack((cats, dogs))
    label_list = np.hstack((label_cats, label_dogs))
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]

    sample_num = len(image_list)
    test_num = ceil(sample_num * test_size)
    train_num = sample_num - test_num
 
    train_images = image_list[0: train_num]
    train_labels = label_list[0: train_num]
    val_images = image_list[train_num: -1]
    val_labels = label_list[train_num: -1]

    return train_images, train_labels, val_images, val_labels


def get_test_files(file_dir):
    """
    Args:
        file_dir: 文件夹路径
    Returns:
        图片的 list
    """
    image_list = []
    for file in os.listdir(file_dir):
        image_list.append(file_dir + file)
    print('There are {} pictures.'.format(len(image_list)))
    return image_list


def get_batch(image, label, image_W, image_H, batch_size, capacity):
    """
    Args:
        image: 图片的 list
        label: label 的 list
        image_W: 图片的宽度
        image_H: 图片的高度
        batch_size: 每个 batch 的图片数
        capacity: 队列的容量
    Returns:
        图片和标签的 batch
    """
    # tf.cast(x, dtype, name=None): Casts a tensor to a new type.
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # 创建队列
    input_queue = tf.train.slice_input_producer([image, label],
                                                num_epochs=None,
                                                shuffle=True,
                                                seed=None,
                                                capacity=capacity)
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)  # 统一图片大小，裁剪/填充
    image = tf.cast(image, tf.float32)
    image = tf.image.per_image_standardization(image)   # 将整幅图片标准化，加速神经网络的训练
    label = input_queue[1]
    
    # 得到 batch
    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=16,
                                              capacity=capacity)
    label_batch = tf.reshape(label_batch, [batch_size])
    return image_batch, label_batch


def get_test_batch(image, image_W, image_H, batch_size, capacity):
    """
    Args:
        image: 图像的 list
        image_W: 图片的宽度
        image_H: 图片的高度
        batch_size: 每个 batch 的图片数
        capacity: 队列的容量
    Returns:
        图片的 batch
    """
    
    image = tf.cast(image, tf.string)

    # 创建队列
    input_queue = tf.train.slice_input_producer([image],
                                                num_epochs=None,
                                                shuffle=True,
                                                seed=None,
                                                capacity=capacity)
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)  # 统一图片大小，裁剪/填充
    image = tf.cast(image, tf.float32)
    image = tf.image.per_image_standardization(image)   # 将整幅图片标准化，加速神经网络的训练
    
    # 得到 batch
    image_batch = tf.train.batch([image],
                                batch_size=batch_size,
                                num_threads=16,
                                capacity=capacity)
    return image_batch


def test():
    """
    功能测试
    """
    import matplotlib.pyplot as plt
    BATCH_SIZE = 8
    CAPACITY = 256
    IMG_W = 208
    IMG_H = 208
    TRAIN_DIR = './data/train/'
    image_list, label_list = get_train_files(TRAIN_DIR)
    image_batch, label_batch = get_train_batch(image_list, label_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)

    with tf.Session() as sess:
        i = 0
        img_num = 1
        coord = tf.train.Coordinator()  # 线程协调器
        threads = tf.train.start_queue_runners(coord=coord)  # Starts all queue runners collected in the graph.
        try:
            while not coord.should_stop() and i < 1:
                img, label = sess.run([image_batch, label_batch])
                for j in np.arange(BATCH_SIZE):
                    print('label: {}'.format(label[j]))
                    plt.figure()
                    plt.imshow(img[j, :, :, :])
                    plt.colorbar()
                    plt.grid(False)
                    plt.savefig('{}.png'.format(img_num))
                    img_num += 1
                i += 1
        except tf.errors.OutOfRangeError:
            print("Test Done.")
        finally:
            coord.request_stop()  # 发出终止所有线程的命令
        coord.join(threads)  # 等待 threads 结束


if __name__ == '__main__':
    # test()
    get_train_files('./data/train')