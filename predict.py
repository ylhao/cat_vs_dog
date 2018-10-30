#encoding: utf-8
import time
from input_data import *
from model import *
import matplotlib.pyplot as plt


def predict():
    N_CLASSES = 2
    IMG_W = 208
    IMG_H = 208
    BATCH_SIZE = 1
    CAPACITY = 256
    TEST_DIR = './data/test/'
    LOGS_DIR = './logs/'
    RESULT_PATH = './res.txt'
    
    sess = tf.Session()
    image_test_list = get_test_files(TEST_DIR)
    image_test_batch = get_test_batch(image_test_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
    train_logits = interface(image_test_batch, N_CLASSES)
    
    '''
    用softmax转化为百分比数值
    '''
    train_logits = tf.nn.softmax(train_logits)

    '''
    载入检查点
    '''
    saver = tf.train.Saver()
    print('\n载入检查点...')
    ckpt = tf.train.get_checkpoint_state(LOGS_DIR)
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('载入成功[global_step = %s]\n' % global_step)
    else:
        print('没有找到检查点，载入失败')
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    fout = open(RESULT_PATH, 'w')
    try:
        for step in range(len(image_test_list)):
            if coord.should_stop():
                break
            prediction = sess.run([train_logits])
            max_index = np.argmax(prediction)
            fout.write('{},{}\n'.format(prediction[0][0][0], prediction[0][0][1]))
            if max_index == 0:
                label = '%.2f%% is a cat.' % (prediction[0][0][0] * 100)
            else:
                label = '%.2f%% is a dog.' % (prediction[0][0][1] * 100)
            print(label)
    except tf.errors.OutOfRangeError:
        print('Done.')
    finally:
        fout.close()
        coord.request_stop()
    coord.join(threads=threads)
    sess.close()


if __name__ == '__main__':
    # training()
    predict()