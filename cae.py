import tensorflow as tf
import numpy as np
import cv2


class CAE:

    lr = 0.001
    lr_decay = 0.99
    opim_func = None
    cost = None
    def __init__(self, inp_sz_, outp_sz_):
        self.inp_sz =inp_sz_
        self.outp_sz = outp_sz_
        self.im_in = tf.placeholder('float',[None,inp_sz_,inp_sz_])
        self.im_out = tf.placeholder('float',[None,outp_sz_,outp_sz_])
        self.im_out = tf.placeholder('float',[None,outp_sz_,outp_sz_])

        self.model = self.autoEncoder(self.im_in)
        self.sess =tf.Session()

        self.lossFunction(self.model)
        self.sess.run(tf.global_variables_initializer())
        # self.saver = tf.save

    def autoEncoder(self, inp_):

        inp = tf.reshape(inp_, shape=[-1,self.inp_sz,self.inp_sz,1])
        #----------------Encoder-----------------------------------#
        kernel_1 = tf.Variable(tf.random_normal([9,9,1,16]))
        kernel_2 = tf.Variable(tf.random_normal([9, 9, 16, 32]))
        kernel_3 = tf.Variable(tf.random_normal([3, 3, 32, 64]))
        # kernel_4 = tf.Variable(tf.random_normal([9, 9, 16, 1]))

        stride_1 = [1,1,1,1]
        stride_2 = [1, 1]
        cLayer1 = tf.nn.conv2d(inp,kernel_1,[1,1,1,1],'VALID')
        cLayer1 = tf.nn.relu(cLayer1)
        cLayer1 = tf.nn.max_pool(cLayer1,ksize=[1,2,2,1], strides=[1,1,1,1], padding='SAME')

        cLayer2 = tf.nn.conv2d(cLayer1, kernel_2, stride_1, 'VALID')
        cLayer2 = tf.nn.relu(cLayer2)
        cLayer2= tf.nn.max_pool(cLayer2,ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

        cLayer3 = tf.nn.conv2d(cLayer2, kernel_3, stride_1, 'VALID')
        encoded = tf.nn.relu(cLayer3)
        encoded = tf.nn.max_pool(cLayer3, ksize=[1, 2, 2, 1], strides=[1, 2,2, 1], padding='SAME')


        print('encoded ',encoded.shape)
        # cLayer3 = tf.image.resize_images(cLayer2, (52, 52))
        # cLayer3 = tf.nn.conv2d(cLayer3, kernel_3, stride_1, 'VALID')

        #----------------Decoder-----------------------------------#
        cLayer4 = tf.layers.conv2d_transpose(inputs=encoded, filters=32, kernel_size=3, strides=(2,2),padding='valid')
        print('clayer 4 ', cLayer4)
        cLayer4 = tf.nn.relu(cLayer4)

        cLayer5 = tf.layers.conv2d_transpose(inputs = cLayer2,   filters=16,kernel_size=9, strides=(2,2),padding='valid')
        print('clayer 5 ',cLayer5)
        cLayer5 = tf.nn.relu(cLayer5)

        cLayer6 = tf.layers.conv2d_transpose(inputs=cLayer5, filters=1, kernel_size=9, strides=(1,1),padding='valid')
        # cLayer6 = tf.nn.softmax(cLayer6)# cLayer4 = tf.nn.max_pool(cLayer4, ksize=(2, 2), strides=(2, 2), padding='same')
        print('cLayer6 ', cLayer6.shape)
        output = tf.reshape(cLayer6,shape=[-1,self.outp_sz,self.outp_sz])
        return  output

    def lossFunction(self, model):

        self.cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=model, labels = self.im_out)
        loss = tf.reduce_mean(self.cost)
        self.opim_func = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def train(self, data_in, data_out):
        data_out = np.array(data_out)
        data_in = np.array(data_in)
        _,c = self.sess.run([self.opim_func,self.cost],feed_dict={self.im_in:data_in, self.im_out:data_out})
        print('cost',np.sum(np.sum(c)))
        self.lr *= self.lr_decay
        print('new learnign rate',self.lr)


