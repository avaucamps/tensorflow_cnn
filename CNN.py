import os
from pathlib import Path
import tensorflow as tf
from DatasetGenerator import DatasetGenerator
import numpy as np
from test_helper import number_of_testing_images, get_labels, create_submission_file


class CNN():
    def __init__(self, 
        train_data_path, 
        valid_data_path, 
        learning_rate, 
        image_size, 
        batch_size, 
        n_classes, 
        base_path,
        n_training_files,
        n_validation_files,
        test_data_path=None):
        self.train_data_path = train_data_path
        self.valid_data_path = valid_data_path
        self.test_data_path = test_data_path
        self.learning_rate = learning_rate 
        self.image_size = image_size
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.base_path = base_path
        self.n_training_files = n_training_files
        self.n_validation_files = n_validation_files

        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self.training = False
        self.checkpoint_dir = base_path + '/checkpoints'
        convnet_path = '/checkpoints/cnn'
        Path(self.checkpoint_dir).mkdir(exist_ok=True)
        self.save_path = base_path + convnet_path


    def build_for_training(self):
        self.set_training_data()
        self.set_inference()
        self.set_loss()
        self.set_optimizer()
        self.set_validation()
        self.set_summary()


    def build_for_testing(self):
        self.set_testing_data()
        self.set_inference()

    
    def set_training_data(self):
        with tf.name_scope('training_data'):
            self.train_data = DatasetGenerator(
                data_path=self.train_data_path, 
                is_data_augmentation_enabled=True,
                image_size=self.image_size, 
                image_channels=3,
                batch_size=self.batch_size
            ).get_data()
            self.valid_data = DatasetGenerator(
                data_path=self.valid_data_path,
                is_data_augmentation_enabled=True,
                image_size=self.image_size,
                image_channels=3,
                batch_size=self.batch_size
            ).get_data()

            iterator = tf.data.Iterator.from_structure(
                self.train_data.output_types, 
                self.train_data.output_shapes
            )
            self.image_input, self.expected_output = iterator.get_next()

            self.train_initializer = iterator.make_initializer(self.train_data)
            self.valid_initializer = iterator.make_initializer(self.valid_data)

        
    def set_testing_data(self):
        with tf.name_scope("data"):
            data = DatasetGenerator(
                data_path=self.test_data_path, 
                is_data_augmentation_enabled=False, 
                image_size=self.image_size,
                image_channels=3,
                batch_size=1
            ).get_test_data()

            iterator = tf.data.Iterator.from_structure(
                data.output_types, 
                data.output_shapes
            )
            self.image_input = iterator.get_next()
            self.test_initializer = iterator.make_initializer(data)


    def set_inference(self):
        with tf.name_scope('CNN'):
            conv1 = tf.layers.conv2d(
                inputs=self.image_input,
                filters=32,
                kernel_size=[5, 5],
                padding='SAME',
                activation=tf.nn.relu,
                name='conv1'
            )
            pool1 = tf.layers.max_pooling2d(
                inputs=conv1,
                pool_size=[2, 2],
                strides=2,
                name='pool1'
            )

            conv2 = tf.layers.conv2d(
                inputs=pool1,
                filters=64,
                kernel_size=[5, 5],
                padding='SAME',
                activation=tf.nn.relu,
                name='conv2'
            )
            pool2 = tf.layers.max_pooling2d(
                inputs=conv2,
                pool_size=[2,2],
                strides=2,
                name='pool2'
            )

            conv3 = tf.layers.conv2d(
                inputs=pool2,
                filters=128,
                kernel_size=[3,3],
                padding='SAME',
                activation=tf.nn.relu,
                name='conv3'
            )
            pool3 = tf.layers.max_pooling2d(
                inputs=conv3,
                pool_size=[2,2],
                strides=2,
                name='pool3'
            )

            feature_dimension = pool3.shape[1] * pool3.shape[2] * pool3.shape[3]
            pool3 = tf.reshape(pool3, [-1, feature_dimension])
            
            fully_connected = tf.layers.dense(pool3, 1024, activation=tf.nn.relu, name='fully_connected')
            dropout = tf.layers.dropout(
                inputs=fully_connected,
                rate=0.35,
                training=self.training,
                name='dropout'
            )
            self.logits = tf.layers.dense(dropout, self.n_classes, name='logits')

    def set_loss(self):
        with tf.name_scope('loss'):
            entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.expected_output, logits=self.logits)
            self.loss = tf.reduce_mean(entropy, name='loss')


    def set_optimizer(self):
        with tf.name_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(
                self.loss,
                global_step=self.global_step
            )

    
    def set_summary(self):
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)
            tf.summary.histogram('histogram_loss', self.loss)
            self.summary_operator = tf.summary.merge_all()


    def set_validation(self):
        with tf.name_scope('predict'):
            predictions = tf.nn.softmax(self.logits)
            correct_predictions = tf.equal(tf.argmax(predictions, 1), tf.argmax(self.expected_output, 1))
            self.accuracy = tf.reduce_sum(tf.cast(correct_predictions, tf.float32))


    def train_epoch(self, epoch, session, saver, writer, step, n_epochs):
        session.run(self.train_initializer)
        self.training = True
        total_epoch_loss = 0
        n_batches = int(self.n_training_files / self.batch_size)
        if (self.n_training_files % self.batch_size) != 0:
            n_batches += 1

        print('Epoch {0}/{1} ##########'.format(epoch, n_epochs))
        for batch_index in range(n_batches):
            _, loss, summaries = session.run([self.optimizer, self.loss, self.summary_operator])
            writer.add_summary(summaries, global_step=step)
            step += 1
            total_epoch_loss += loss
            print('Batch {0}/{1}: loss = {2}'.format(batch_index, n_batches, loss))

        saver.save(session, self.save_path, step)

        average_epoch_loss = total_epoch_loss/n_batches
        print('Epoch {0}/{1}: average loss = {2} ##########'.format(epoch, n_epochs, average_epoch_loss))    

        return step


    def valid_epoch(self, epoch, session, writer, step, n_epochs):
        session.run(self.valid_initializer)
        self.training = False
        total_correct_predictions = 0
        n_batches = int(self.n_validation_files / self.batch_size)
        if (self.n_validation_files % self.batch_size) != 0:
            n_batches += 1

        for _ in range(n_batches):
            batch_accuracy, summaries = session.run([self.accuracy, self.summary_operator])
            writer.add_summary(summaries, global_step=step)
            total_correct_predictions += batch_accuracy

        epoch_accuracy = total_correct_predictions / self.n_validation_files
        print('Epoch {0}/{1}: accuracy = {2}'.format(epoch, n_epochs, epoch_accuracy))
        

    def train(self, n_epochs):
        self.build_for_training()
        writer = tf.summary.FileWriter(self.base_path + '/graphs', tf.get_default_graph())

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            step = self.global_step.eval()

            for epoch in range(10, 20):
                step = self.train_epoch(epoch+1, sess, saver, writer, step, n_epochs)
                self.valid_epoch(epoch+1, sess, writer, step, n_epochs)

            saver.save(sess, self.save_path)
        writer.close()


    def test(self, submission_filename=None):
        self.training = False
        self.build_for_testing()

        n_test_images = number_of_testing_images(self.test_data_path)
        labels = get_labels(self.train_data_path)
        predictions = np.empty([n_test_images, len(labels)])
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(self.test_initializer)
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            for i in range(n_test_images):
                predictions[i] = sess.run(self.logits)

        if submission_filename:
            create_submission_file(predictions, submission_filename, labels, self.test_data_path)