#encoding: utf-8
import time
from input_data import *
from model import *
import matplotlib.pyplot as plt


def training():
    N_CLASSES = 2
    IMG_W = 208
    IMG_H = 208
    BATCH_SIZE = 64
    CAPACITY = 256
    MAX_STEP = 15000
    LEARNING_RATE = 1e-4
    TRAIN_DIR = './data/train/'
    LOGS_DIR = './logs/'
    TRAIN_LOGS_DIR = './logs/train/'
    VAL_LOGS_DIR = './logs/val/'

    sess = tf.Session()

    train_image_list, train_label_list, val_image_list, val_label_list = get_files(TRAIN_DIR)
    train_image_batch, train_label_batch = get_batch(train_image_list, train_label_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
    val_image_batch, val_label_batch = get_batch(val_image_list, val_label_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
    logits = interface(train_image_batch, N_CLASSES)
    loss = losses(logits, train_label_batch)
    acc = evaluation(logits, train_label_batch)
    train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
    
    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])
    y_ = tf.placeholder(tf.int16, shape=[BATCH_SIZE])

    '''
    统计参数数目
    '''
    var_list = tf.trainable_variables()
    paras_count = tf.reduce_sum([tf.reduce_prod(v.shape) for v in var_list])
    print('参数数目: {}'.format(sess.run(paras_count)))

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    summary_op = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(TRAIN_LOGS_DIR, sess.graph)
    val_writer = tf.summary.FileWriter(VAL_LOGS_DIR, sess.graph)

    s_t = time.time()
    
    try:
        for step in range(MAX_STEP):
            if coord.should_stop():
                break
            train_images, train_labels = sess.run([train_image_batch, train_label_batch])
            _, train_loss, train_acc = sess.run([train_op, loss, acc], feed_dict={x: train_images, y_: train_labels})

            '''
            实时记录训练过程并显示
            '''
            if step % 100 == 0 or (step + 1) == MAX_STEP:
                val_images, val_labels = sess.run([val_image_batch, val_label_batch])
                val_loss, val_acc = sess.run([loss, acc], feed_dict={x: val_images, y_: val_labels})
                runtime = time.time() - s_t

                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)

                summary_str = sess.run(summary_op)
                val_writer.add_summary(summary_str, step)

                print('Step: %6d, train loss: %.8f, train accuracy: %.2f%%, val loss = %.8f, val accuracy = %.2f%%, time:%.2fs, time left: %.2fhours'
                      % (step, train_loss, train_acc * 100, val_loss, val_acc * 100, runtime, (MAX_STEP - step) * runtime / 360000))
                s_t = time.time()

            '''
            保存检查点
            '''
            if step % 1000 == 0 or step == MAX_STEP - 1:
                checkpoint_path = os.path.join(LOGS_DIR, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

    except tf.errors.OutOfRangeError:
        print('Done.')
    
    finally:
        coord.request_stop()
    
    coord.join(threads=threads)
    sess.close()


if __name__ == '__main__':
    training()