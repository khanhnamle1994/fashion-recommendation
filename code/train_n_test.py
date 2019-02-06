'''
This is the main training profile.
'''
from fashion_input import *
import os
import tensorflow as tf
import time
from datetime import datetime
from simple_resnet import *
from hyper_parameters import *

TRAIN_DIR = 'logs_' + FLAGS.version + '/'
TRAIN_LOG_PATH = FLAGS.version + '_error.csv'

REPORT_FREQ = 50
TRAIN_BATCH_SIZE = 32
VALI_BATCH_SIZE = 25
TEST_BATCH_SIZE = 25
FULL_VALIDATION = False
Error_EMA = 0.98

STEP_TO_TRAIN = 45000
DECAY_STEP0 = 25000
DECAY_STEP1 = 35000

def generate_validation_batch(df):
    '''
    :param df: a pandas dataframe with validation image paths and the corresponding labels
    :return: two random numpy arrays: validation_batch and validation_label
    '''
    offset = np.random.choice(len(df) - VALI_BATCH_SIZE, 1)[0]
    validation_df = df.iloc[offset:offset+VALI_BATCH_SIZE, :]

    validation_batch, validation_label, validation_bbox_label = load_data_numpy(validation_df)
    return validation_batch, validation_label, validation_bbox_label


class Train:
    '''
    The class defining the training process and relevant helper functions
    '''
    def __init__(self):
        self.placeholders()

    def loss(self, logits, bbox, labels, bbox_labels):
        labels = tf.cast(labels, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross_entropy_per_example')
        mse_loss = tf.reduce_mean(tf.losses.mean_squared_error(bbox_labels, bbox), name='mean_square_loss')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        return cross_entropy_mean + mse_loss

    def top_k_error(self, predictions, labels, k):
        batch_size = predictions.get_shape().as_list()[0]
        in_top1 = tf.to_float(tf.nn.in_top_k(predictions, labels, k=1))
        num_correct = tf.reduce_sum(in_top1)
        return (batch_size - num_correct) / float(batch_size)


    def placeholders(self):
        self.image_placeholder = tf.placeholder(dtype=tf.float32, shape=[TRAIN_BATCH_SIZE,
                                                                        IMG_ROWS, IMG_COLS, 3])
        self.label_placeholder = tf.placeholder(dtype=tf.int32, shape=[TRAIN_BATCH_SIZE])
        self.bbox_placeholder = tf.placeholder(dtype=tf.float32, shape=[TRAIN_BATCH_SIZE, 4])

        self.vali_image_placeholder = tf.placeholder(dtype=tf.float32, shape=[VALI_BATCH_SIZE,
                                                                IMG_ROWS, IMG_COLS, 3])
        self.vali_label_placeholder = tf.placeholder(dtype=tf.int32, shape=[VALI_BATCH_SIZE])
        self.vali_bbox_placeholder = tf.placeholder(dtype=tf.float32, shape=[VALI_BATCH_SIZE, 4])

        self.lr_placeholder = tf.placeholder(dtype=tf.float32, shape=[])
        self.dropout_prob_placeholder = tf.placeholder(dtype=tf.float32, shape=[])


    def train_operation(self, global_step, total_loss, top1_error):
        tf.summary.scalar('learning_rate', self.lr_placeholder)
        tf.summary.scalar('train_loss', total_loss)
        tf.summary.scalar('train_top1_error', top1_error)

        ema = tf.train.ExponentialMovingAverage(0.95, global_step)
        train_ema_op = ema.apply([total_loss, top1_error])
        tf.summary.scalar('train_top1_error_avg', ema.average(top1_error))
        tf.summary.scalar('train_loss_avg', ema.average(total_loss))

        opt = tf.train.MomentumOptimizer(learning_rate=self.lr_placeholder, momentum=0.9)
        train_op = opt.minimize(total_loss, global_step=global_step)
        return train_op, train_ema_op


    def validation_op(self, validation_step, top1_error, loss):
        ema = tf.train.ExponentialMovingAverage(0.0, validation_step)
        ema2 = tf.train.ExponentialMovingAverage(0.95, validation_step)
        val_op = tf.group(validation_step.assign_add(1), ema.apply([top1_error, loss]), ema2.apply([top1_error, loss]))
        top1_error_val = ema.average(top1_error)
        top1_error_avg = ema2.average(top1_error)
        loss_val = ema.average(loss)
        loss_val_avg = ema2.average(loss)
        tf.summary.scalar('val_top1_error', top1_error_val)
        tf.summary.scalar('val_top1_error_avg', top1_error_avg)
        tf.summary.scalar('val_loss', loss_val)
        tf.summary.scalar('val_loss_avg', loss_val_avg)
        return val_op


    def full_validation(self, validation_df, sess, vali_loss, vali_top1_error, batch_data, batch_label, batch_bbox):
        num_batches = len(validation_df) // VALI_BATCH_SIZE
        error_list = []
        loss_list = []

        for i in range(num_batches):
            offset = i * VALI_BATCH_SIZE
            vali_batch_df = validation_df.iloc[offset:offset+VALI_BATCH_SIZE, :]
            validation_image_batch, validation_labels_batch, validation_bbox_batch = load_data_numpy(vali_batch_df)

            vali_error, vali_loss_value = sess.run([vali_top1_error, vali_loss],
                                              {self.image_placeholder: batch_data,
                                                     self.label_placeholder: batch_label,
                                                    self.bbox_placeholder:batch_bbox,
                                                     self.vali_image_placeholder: validation_image_batch,
                                                     self.vali_label_placeholder: validation_labels_batch,
                                                    self.vali_bbox_placeholder: validation_bbox_batch,
                                                     self.lr_placeholder: FLAGS.learning_rate,
                                                     self.dropout_prob_placeholder: 0.5})
            error_list.append(vali_error)
            loss_list.append(vali_loss_value)

        return np.mean(error_list), np.mean(loss_list)



    def train(self):
        train_df = prepare_df(FLAGS.train_path, usecols=['image_path', 'category', 'x1_modified', 'y1_modified', 'x2_modified', 'y2_modified'])
        vali_df = prepare_df(FLAGS.vali_path, usecols=['image_path', 'category', 'x1_modified', 'y1_modified', 'x2_modified', 'y2_modified'])

        num_train = len(train_df)
        global_step = tf.Variable(0, trainable=False)
        validation_step = tf.Variable(0, trainable=False)


        logits, bbox, _ = inference(self.image_placeholder, n=FLAGS.num_residual_blocks, reuse=False,
                                    keep_prob_placeholder=self.dropout_prob_placeholder)
        vali_logits, vali_bbox, _ = inference(self.vali_image_placeholder, n=FLAGS.num_residual_blocks,
                                         reuse=True, keep_prob_placeholder=self.dropout_prob_placeholder)


        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = self.loss(logits, bbox, self.label_placeholder, self.bbox_placeholder)
        full_loss = tf.add_n([loss] + reg_losses)

        predictions = tf.nn.softmax(logits)
        top1_error = self.top_k_error(predictions, self.label_placeholder, 1)
        vali_loss = self.loss(vali_logits, vali_bbox, self.vali_label_placeholder, self.vali_bbox_placeholder)
        vali_predictions = tf.nn.softmax(vali_logits)
        vali_top1_error = self.top_k_error(vali_predictions, self.vali_label_placeholder, 1)


        train_op, train_ema_op = self.train_operation(global_step, full_loss, top1_error)
        val_op = self.validation_op(validation_step, vali_top1_error, vali_loss)

        saver = tf.train.Saver(tf.all_variables())
        summary_op = tf.summary.merge_all()
        init = tf.initialize_all_variables()
        sess = tf.Session()

        if FLAGS.continue_train_ckpt is True:
            print('Model restored!')
            saver.restore(sess, FLAGS.ckpt_path)
        else:
            sess.run(init)
        summary_writer = tf.summary.FileWriter(TRAIN_DIR, sess.graph)

        step_list = []
        train_error_list = []
        vali_error_list = []
        min_error = 0.5

        for step in range(STEP_TO_TRAIN):

            offset = np.random.choice(num_train - TRAIN_BATCH_SIZE, 1)[0]

            train_batch_df = train_df.iloc[offset:offset+TRAIN_BATCH_SIZE, :]
            batch_data, batch_label, batch_bbox = load_data_numpy(train_batch_df)

            vali_image_batch, vali_labels_batch, vali_bbox_batch = generate_validation_batch(vali_df)

            start_time = time.time()

            if step == 0:
                if FULL_VALIDATION is True:
                    top1_error_value, vali_loss_value = self.full_validation(vali_df,
                                                                             sess=sess,
                                                            vali_loss=vali_loss,
                                                            vali_top1_error=vali_top1_error,
                                                            batch_data=batch_data,
                                                            batch_label=batch_label,
                                                            batch_bbox=batch_bbox)
                    vali_summ = tf.Summary()
                    vali_summ.value.add(tag='full_validation_error',
                                    simple_value=top1_error_value.astype(np.float))
                    vali_summ.value.add(tag='full_validation_loss',
                                    simple_value=vali_loss_value.astype(np.float))
                    summary_writer.add_summary(vali_summ, step)
                    summary_writer.flush()

                else:
                    _, top1_error_value, vali_loss_value = sess.run([val_op, vali_top1_error,
                                                                     vali_loss],
                                                    {self.image_placeholder: batch_data,
                                                     self.label_placeholder: batch_label,
                                                     self.vali_image_placeholder: vali_image_batch,
                                                     self.vali_label_placeholder: vali_labels_batch,
                                                     self.lr_placeholder: FLAGS.learning_rate,
                                                     self.bbox_placeholder: batch_bbox,
                                                     self.vali_bbox_placeholder: vali_bbox_batch,
                                                     self.dropout_prob_placeholder: 1.0})
                print('Validation top1 error = %.4f' % top1_error_value)
                print('Validation loss = ', vali_loss_value)
                print('----------------------------')


            _, _, loss_value, train_top1_error = sess.run([train_op, train_ema_op, loss,
                    top1_error], {self.image_placeholder: batch_data,
                                  self.label_placeholder: batch_label,
                                  self.bbox_placeholder: batch_bbox,
                                  self.vali_image_placeholder: vali_image_batch,
                                  self.vali_label_placeholder: vali_labels_batch,
                                  self.vali_bbox_placeholder: vali_bbox_batch,
                                  self.lr_placeholder: FLAGS.learning_rate,
                                  self.dropout_prob_placeholder: 0.5})
            duration = time.time() - start_time

            if step % REPORT_FREQ == 0:
                summary_str = sess.run(summary_op, {self.image_placeholder: batch_data,
                                                    self.label_placeholder: batch_label,
                                                    self.bbox_placeholder: batch_bbox,
                                                    self.vali_image_placeholder: vali_image_batch,
                                                    self.vali_label_placeholder: vali_labels_batch,
                                                    self.vali_bbox_placeholder: vali_bbox_batch,
                                                    self.lr_placeholder: FLAGS.learning_rate,
                                                    self.dropout_prob_placeholder: 0.5})
                summary_writer.add_summary(summary_str, step)


                num_examples_per_step = TRAIN_BATCH_SIZE
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = %.4f (%.1f examples/sec; %.3f ' 'sec/batch)')
                print (format_str % (datetime.now(), step, loss_value, examples_per_sec, sec_per_batch))
                print('Train top1 error = ', train_top1_error)

                if FULL_VALIDATION is True:
                    top1_error_value, vali_loss_value = self.full_validation(vali_df,
                                                                             sess=sess,
                                                            vali_loss=vali_loss,
                                                            vali_top1_error=vali_top1_error,
                                                            batch_data=batch_data,
                                                            batch_label=batch_label,
                                                            batch_bbox=batch_bbox)
                    vali_summ = tf.Summary()
                    vali_summ.value.add(tag='full_validation_error',
                                    simple_value=top1_error_value.astype(np.float))
                    vali_summ.value.add(tag='full_validation_loss',
                                    simple_value=vali_loss_value.astype(np.float))
                    summary_writer.add_summary(vali_summ, step)
                    summary_writer.flush()

                else:

                    _, top1_error_value, vali_loss_value = sess.run([val_op, vali_top1_error,
                                                                 vali_loss],
                                                {self.image_placeholder: batch_data,
                                                 self.label_placeholder: batch_label,
                                                 self.bbox_placeholder: batch_bbox,
                                                 self.vali_image_placeholder: vali_image_batch,
                                                 self.vali_label_placeholder: vali_labels_batch,
                                                 self.vali_bbox_placeholder: vali_bbox_batch,
                                                 self.lr_placeholder: FLAGS.learning_rate,
                                                 self.dropout_prob_placeholder: 0.5})

                print('Validation top1 error = %.4f' % top1_error_value)
                print('Validation loss = ', vali_loss_value)
                print('----------------------------')

                if top1_error_value < min_error:
                    min_error = top1_error_value
                    checkpoint_path = os.path.join(TRAIN_DIR, 'min_model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)
                    print('Current lowest error = ', min_error)

                step_list.append(step)
                train_error_list.append(train_top1_error)
                vali_error_list.append(top1_error_value)


            if step == DECAY_STEP0 or step == DECAY_STEP1:
                FLAGS.learning_rate = FLAGS.learning_rate * 0.1


            if step % 10000 == 0 or (step + 1) == STEP_TO_TRAIN:
                checkpoint_path = os.path.join(TRAIN_DIR, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

                error_df = pd.DataFrame(data={'step':step_list, 'train_error':
                    train_error_list, 'validation_error': vali_error_list})
                error_df.to_csv(TRAIN_DIR + TRAIN_LOG_PATH, index=False)

            if (step + 1) == STEP_TO_TRAIN:
                checkpoint_path = os.path.join(TRAIN_DIR, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

                error_df = pd.DataFrame(data={'step': step_list, 'train_error':
                    train_error_list, 'validation_error': vali_error_list})
                error_df.to_csv(TRAIN_DIR + TRAIN_LOG_PATH, index=False)

        print('Training finished!!')

    def test(self):
        self.test_image_placeholder = tf.placeholder(dtype=tf.float32, shape=[25, IMG_ROWS,
                                                                              IMG_COLS, 3])
        self.test_label_placeholder = tf.placeholder(dtype=tf.int32, shape=[25])

        ##########################
        # Build test graph
        logits, global_pool = inference(self.test_image_placeholder, n=FLAGS.num_residual_blocks, reuse=False,
                                              keep_prob_placeholder=self.dropout_prob_placeholder)
        predictions = tf.nn.softmax(logits)
        test_error = self.top_k_error(predictions, self.test_label_placeholder, 1)

        saver = tf.train.Saver(tf.all_variables())
        sess = tf.Session()
        saver.restore(sess, FLAGS.test_ckpt_path)
        print('Model restored!')
        ##########################

        test_df = prepare_df(FLAGS.test_path, usecols=['image_path', 'category', 'x1', 'y1', 'x2', 'y2'], shuffle=False)
        test_df = test_df.iloc[-25:, :]

        prediction_np = np.array([]).reshape(-1, 6)
        fc_np = np.array([]).reshape(-1, 64)
        # Hack here: 25 as batch size. 50000 images in total
        for step in range(len(test_df) // TEST_BATCH_SIZE):
            if step % 100 == 0:
                print('Testing %i batches...' %step)
                if step != 0:
                    print('Test_error = ', test_error_value)

            df_batch = test_df.iloc[step*25 : (step+1)*25, :]
            test_batch, test_label = load_data_numpy(df_batch)

            prediction_batch_value, test_error_value = sess.run([predictions, test_error],
                                                               feed_dict={
                self.test_image_placeholder:test_batch, self.test_label_placeholder: test_label})
            fc_batch_value = sess.run(global_pool, feed_dict={
                self.test_image_placeholder:test_batch, self.test_label_placeholder: test_label})

            prediction_np = np.concatenate((prediction_np, prediction_batch_value), axis=0)
            fc_np = np.concatenate((fc_np, fc_batch_value))

        print('Predictin array has shape ', fc_np.shape)
        np.save(FLAGS.fc_path, fc_np[-5:,:])

train = Train()
train.train()
train.test()
