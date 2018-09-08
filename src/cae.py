import tensorflow as tf
import numpy as np
import cv2
import os

class CAE:

    lr = 0.001
    lr_decay = 0.99
    opim_func = None
    cost = None
    save_path = None
    def __init__(self, inp_sz_, outp_sz_, save_path ='',show_test_ = False):
        self.show_test = show_test_
        self.inp_sz =inp_sz_
        self.outp_sz = outp_sz_
        self.classes = 2
        self.im_in = tf.placeholder('float',[None,inp_sz_,inp_sz_])

        self.im_out = tf.placeholder('float',[None,outp_sz_,outp_sz_,self.classes])

        self.model = self.autoEncoder(self.im_in)
        self.sess =tf.Session()

        self.lossFunction(self.model)
        self.sess.run(tf.global_variables_initializer())

        self.ini_saver(save_path)

    def ini_saver(self,save_path):

        self.saver = tf.train.Saver()

        if save_path is not None and os.path.isfile(os.path.abspath(save_path + '.meta')):

            self.load_path = save_path

            try:
                self.saver.restore(self.sess, self.load_path)
                print('model loaded')
            except:
                print('could not reload weights')

        if save_path is not None:
            self.save_path = save_path


    def autoEncoder(self, inp_):

        inp = tf.reshape(inp_, shape=[-1,self.inp_sz,self.inp_sz,1])
        #----------------Encoder-----------------------------------#
        kernel_1 = tf.Variable(tf.random_normal([9,9,1,16]))
        kernel_1_2 = tf.Variable(tf.random_normal([9, 9, 16, 16]))
        kernel_2 = tf.Variable(tf.random_normal([9, 9, 16, 32]))
        kernel_2_2 = tf.Variable(tf.random_normal([9, 9, 32, 32]))
        kernel_3 = tf.Variable(tf.random_normal([3, 3, 32, 64]))
        kernel_3_2 = tf.Variable(tf.random_normal([3, 3, 64, 64]))
        kernel_fc = tf.Variable(tf.random_normal([1, 1, 64, 64]))

        stride_1 = [1,1,1,1]
        stride_2 = [1, 1]

        encode_1 = tf.nn.conv2d(inp,kernel_1,[1,1,1,1],'VALID')
        encode_1 = tf.nn.relu(encode_1)
        # layer_1 = tf.contrib.layers.batch_norm(layer_1, data_format='NHWC', center=True, scale=True,is_training=training)
        encode_1_2 = tf.nn.conv2d(encode_1, kernel_1_2, [1, 1, 1, 1], 'SAME')
        encode_1_2 = tf.nn.relu(encode_1_2)
        encode_1_2 = tf.nn.max_pool(encode_1_2,ksize=[1,2,2,1], strides=[1,1,1,1], padding='SAME')

        encode_2 = tf.nn.conv2d(encode_1_2, kernel_2, stride_1, 'VALID')
        encode_2 = tf.nn.relu(encode_2)
        encode_2_2 = tf.nn.conv2d(encode_2, kernel_2_2, [1, 1, 1, 1], 'SAME')
        encode_2_2 = tf.nn.relu(encode_2_2)
        encode_2_2= tf.nn.max_pool(encode_2_2,ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

        encode_3 = tf.nn.conv2d(encode_2_2, kernel_3, stride_1, 'VALID')
        encode_3 = tf.nn.relu(encode_3)
        encode_3_2 = tf.nn.conv2d(encode_3, kernel_3_2, [1, 1, 1, 1], 'SAME')
        encode_3_2 = tf.nn.relu(encode_3_2)

        encoded_3_maxpool = tf.nn.max_pool(encode_3_2, ksize=[1, 2, 2, 1], strides=[1, 2,2, 1], padding='SAME')

        connected = tf.nn.conv2d( encoded_3_maxpool, kernel_fc,stride_1, 'SAME')

        print('encoded_3_maxpool ',encoded_3_maxpool.shape)
        conc_enc = tf.concat((encoded_3_maxpool,connected),axis = 3)

        #----------------Decoder-----------------------------------#
        decode_1 = tf.layers.conv2d_transpose(inputs=conc_enc, filters=32, kernel_size=3, strides=(2,2),padding='valid')
        decode_1 = tf.nn.relu(decode_1)
        decode_1_2 = tf.layers.conv2d(decode_1,32,3,(1,1),'same')
        decode_1_2 = tf.nn.tanh(decode_1_2)
        print('decode 1 ', decode_1)

        conc_enc_2 = tf.concat((encode_2_2, decode_1_2), axis=3)
        decode_2 = tf.layers.conv2d_transpose(inputs = conc_enc_2,   filters=16,kernel_size=9, strides=(2,2),padding='valid')
        print('decode 2 ',decode_2)
        decode_2 = tf.nn.relu(decode_2)
        decode_2_2 = tf.layers.conv2d(decode_2, 16, 9, (1, 1), 'same')
        decode_2_2 = tf.nn.tanh(decode_2_2)

        conc_enc_3 = tf.concat((encode_1_2, decode_2_2), axis=3)
        decode_3 = tf.layers.conv2d_transpose(inputs=conc_enc_3, filters=9, kernel_size=9, strides=(1,1),padding='valid')
        decode_3 = tf.nn.relu(decode_3)
        decode_3_2 = tf.layers.conv2d(decode_3, 2, 9, (1, 1), 'same')
        print('decode_3_2 ', decode_3_2.shape)
        output = tf.reshape(decode_3_2,shape=[-1,self.outp_sz,self.outp_sz,self.classes])

        return output


    def lossFunction(self, model):

        self.cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=model, labels = self.im_out)
        loss = tf.reduce_mean(self.cost)
        self.opim_func = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def train(self, data_in, data_out):
        data_out = np.array(data_out)
        data_in = np.array(data_in)

        if len(data_in) is not len(data_out) or len(data_in) < 2:
            return

        _,c = self.sess.run([self.opim_func,self.cost],feed_dict={self.im_in:data_in, self.im_out:data_out})
        print('cost',np.sum(np.sum(c)))
        self.lr *= self.lr_decay

        # print('new learnign rate',self.lr)

        self.test(data_in[10],data_out[10])

        if self.save_path is not None:
            save_path = self.saver.save(self.sess, self.save_path)
            print("Model saved in file: %s" % save_path)


    def test(self, data_in, expected_out):

        if not self.show_test:
            return

        input = np.array(data_in)
        input = np.reshape(input, [1,self.inp_sz,self.inp_sz])
        out = self.sess.run(tf.nn.softmax(self.model),feed_dict={self.im_in:input})

        zeros = np.zeros(shape=[self.inp_sz, self.inp_sz, 1])
        gray_scale_out = np.reshape(out, [self.inp_sz, self.inp_sz, self.classes])
        gray_scale_out = np.array(gray_scale_out*255, dtype=np.uint8)
        result = np.concatenate((gray_scale_out,zeros),axis=2)

        gray_scale_expect = np.array(expected_out*255, dtype=np.uint8)
        expected = np.concatenate((gray_scale_expect, zeros),axis = 2)
        # gray_scale_out = np.reshape(gray_scale_out,[self.inp_sz, self.inp_sz,self.classes])
        im_result = np.concatenate(( expected, result),axis=1)
        cv2.imshow('result ',im_result)
        cv2.imshow('data_in',data_in)
        cv2.waitKey(100)


