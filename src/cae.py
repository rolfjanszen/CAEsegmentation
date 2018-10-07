import os

import cv2
import numpy as np
import tensorflow as tf


class CAE:

    lr = 0.002
    lr_decay = 0.99
    opim_func = None
    cost = None
    save_path = None
    input_channels = 3
    def __init__(self, inp_sz_, outp_sz_, save_path ='',show_test_ = False,input_channels_=1):
        self.show_test = show_test_
        self.inp_sz =inp_sz_
        self.outp_sz = outp_sz_
        self.classes = 2
        self.im_in = tf.placeholder('float',[None,inp_sz_,inp_sz_,input_channels_])
        self.input_channels = input_channels_
        self.im_out = tf.placeholder('float',[None,outp_sz_,outp_sz_,self.classes])
        self.is_training = tf.placeholder(tf.bool)
        self.model = self.autoEncoder(self.im_in,self.is_training)
        self.sess =tf.Session()
        self.summ_writer = tf.summary.FileWriter(save_path, self.sess.graph)
        self.lossFunction(self.model)
        self.sess.run(tf.global_variables_initializer())

        self.iniSaver(save_path)


    def iniSaver(self,save_path):

        self.saver = tf.train.Saver()

        if save_path is not None and os.path.isfile(os.path.abspath(save_path + '.meta')):

            self.save_path = save_path+'/segment.ckpt'

            try:
                self.saver.restore(self.sess,   self.save_path)
                print('model loaded')
            except:
                print('could not reload weights')
        self.summ_writer = tf.summary.FileWriter(save_path,self.sess.graph)
        # if save_path is not None:
        #     self.save_path = save_path

    def BatchNormLayer(self, input_, filters_, kernel_, do_maxpool_, training,name):


        layerconv = tf.layers.conv2d(inputs=input_,filters = filters_, kernel_size =kernel_,strides =(1, 1), padding='same',activation=None, name=name)

        layerconv = tf.nn.leaky_relu(layerconv)
        if do_maxpool_:
            layerconv_bn = tf.nn.max_pool(layerconv_bn, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        return layerconv

    def autoEncoder(self, inp_, is_training):
        # inp_ = tf.divide(inp_,255)
        inp = tf.reshape(inp_, shape=[-1,self.inp_sz,self.inp_sz,self.input_channels])
        #----------------Encoder-----------------------------------#
        with tf.variable_scope('cnn_kernels'):
            kernel_1 = tf.Variable(tf.random_normal([7,7, self.input_channels,16]))
            kernel_1_2 = tf.Variable(tf.random_normal([5, 5, 16, 16]))
            kernel_2 = tf.Variable(tf.random_normal([3, 3, 16, 32]))
            kernel_2_2 = tf.Variable(tf.random_normal([3, 3, 32, 32]))
            kernel_3 = tf.Variable(tf.random_normal([3, 3, 32, 64]))
            kernel_3_2 = tf.Variable(tf.random_normal([3, 3, 64, 64]))
            kernel_fc = tf.Variable(tf.random_normal([1, 1, 64, 64]))

        stride_1 = [1,1,1,1]

        with tf.name_scope('unet_model') as scope:
            encode_1 = tf.nn.conv2d(inp,kernel_1,[1,1,1,1],'SAME',name='encode1')
            encode_1 = tf.nn.relu(encode_1)
            # encode_1 = self.BatchNormLayer(inp, 16, 7, False,is_training)

            encode_1_2 = tf.nn.conv2d(encode_1, kernel_1_2, [1, 1, 1, 1], 'SAME',name='encode1_2')

            encode_1_2 = tf.nn.relu(encode_1_2)
            # bn_encode_1_2 = tf.contrib.layers.batch_norm(encode_1_2,    center=True, scale=True,is_training=is_training)
            # bn_encode_1_2 = tf.nn.relu(bn_encode_1_2)
            # encode_1_2 = self.BatchNormLayer(encode_1, 16, 3, False, is_training)
            maxpool_encode_1_2 = tf.nn.max_pool(encode_1_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME',name='maxpoolencode1')

            encode_2 = tf.nn.conv2d(maxpool_encode_1_2, kernel_2, stride_1, 'SAME',name='encode2')
            encode_2 = tf.nn.relu(encode_2)
            # encode_2 = self.BatchNormLayer(maxpool_encode_1_2, 32, 3, False,is_training)
            encode_2_2 = tf.nn.conv2d(encode_2, kernel_2_2, [1, 1, 1, 1], 'SAME',name='encode2_2')
            encode_2_2 = tf.nn.relu(encode_2_2)
            maxpool_encode_2_2= tf.nn.max_pool(encode_2_2,ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

            # encode_3 = tf.nn.conv2d(maxpool_encode_2_2, kernel_3, stride_1, 'SAME')
            # encode_3 = tf.nn.relu6(encode_3)
            encode_3 = self.BatchNormLayer(maxpool_encode_2_2, 64, 3, False, is_training,name='encode3')
            encode_3_2 = tf.nn.conv2d(encode_3, kernel_3_2, [1, 1, 1, 1], 'SAME')
            encode_3_2 = tf.nn.relu(encode_3_2)

            encoded_3_maxpool = tf.nn.max_pool(encode_3_2, ksize=[1, 2, 2, 1], strides=[1, 2,2, 1], padding='SAME')

            connected = tf.nn.conv2d( encoded_3_maxpool, kernel_fc,stride_1, 'SAME')

            print('encoded_3_maxpool ',encoded_3_maxpool.shape)

            #----------------Decoder-----------------------------------#
            decode_1 = tf.layers.conv2d_transpose(inputs=connected, filters=64, kernel_size=3, strides=(2,2),padding='same',activation=None)
            print(' graph ',decode_1.graph)
            decode_1 = tf.nn.relu(decode_1)
            conc_decode_1 = tf.concat((encode_3_2, decode_1), axis=3)
            # decode_1_2 = tf.layers.conv2d(conc_decode_1,filters=64, kernel_size=3, strides =(1, 1), padding='same')
            # decode_1_2 = tf.nn.relu6(decode_1_2)
            decode_1_2 = self.BatchNormLayer(conc_decode_1, 32, 3, False, is_training,name='decode1_2')
            print('decode 1 ', decode_1)

            decode_2 = tf.layers.conv2d_transpose(inputs = decode_1_2,   filters=32,kernel_size=3, strides=(2,2),padding='same',activation=None)
            print('decode 2 ',decode_2)
            decode_2 = tf.nn.relu(decode_2)
            conc_enc_2 = tf.concat((encode_2_2, decode_2), axis=3)
            # decode_2_2 = tf.layers.conv2d(conc_enc_2, filters=32, kernel_size=3, strides =(1, 1), padding='same')
            # decode_2_2 = tf.nn.relu6(decode_2_2)
            decode_2_2 = self.BatchNormLayer(conc_enc_2, 32, 3, False, is_training,name='decode2_2')

            decode_3 = tf.layers.conv2d_transpose(inputs=decode_2_2, filters=16, kernel_size=5, strides=(2,2),padding='same',activation=None)
            decode_3 = tf.nn.leaky_relu(decode_3)
            conc_enc_3 = tf.concat((encode_1_2, decode_3), axis=3)

            decode_3_2 = tf.layers.conv2d(inputs =conc_enc_3, filters=2, kernel_size=7, strides =(1, 1), padding='same',activation=None)
            print('decode_3_2 ', decode_3_2.shape)
            # decode_3_2 =tf.nn.sigmoid(decode_3_2)
            output = tf.reshape(decode_3_2,shape=[-1,self.outp_sz,self.outp_sz,self.classes])

        return output


    def lossFunction(self, model):

        sig_model = tf.nn.sigmoid(model)
        sum_mat = tf.add(sig_model[:,:,:,0],self.im_out[:,:,:,0])

        sum_mat = tf.Print(sum_mat, [sum_mat], 'sum_mat')

        area_clipped = tf.clip_by_value(sum_mat,0,1.1)
        area_union  = tf.reduce_sum(area_clipped)
        area_mul = tf.multiply(sig_model[:, :, :, 0], self.im_out[:, :, :, 0])
        area_mul = tf.Print(area_mul,[area_mul],'area_mul')
        area_overlap =  tf.reduce_sum(area_mul)
        area_overlap = tf.Print(area_overlap, [area_overlap], 'area_overlap')

        iou =(1 - area_overlap/area_union)*100
        #
        iou= tf.Print(iou,[iou], 'iou')
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=model, labels = self.im_out)
        # iou = tf.metrics.mean_iou(predictions=model, labels=self.im_out,num_classes=self.input_channels)
        # iou_sum = tf.reduce_sum(iou)
        tf.summary.scalar(name = 'iou', tensor=iou)

        self.cost = tf.reduce_mean(loss)
        tf.summary.scalar(name='loss', tensor=self.cost)

        optimize_function = tf.train.AdamOptimizer(self.lr)
        self.opim_func = optimize_function.minimize(self.cost)
        grads = optimize_function.compute_gradients(self.cost)

        gradients =[]
        for g in grads:
            print(g)
            if(g[0] != None and g[1] != None):
                gradients.append(tf.summary.histogram("%s-grad" % g[1].name, g[0]))
        tf.summary.merge(gradients)
        # grad_vals = sess.run(fetches=grad_summ_op, feed_dict=feed_dict)
        # writer['train'].add_summary(grad_vals)

        self.merged = tf.summary.merge_all()

    def train(self, data_in, data_out, step):
        # data_out = np.array(data_out)
        # data_in = np.array(data_in)

        if len(data_in) is not len(data_out) or len(data_in) < 2:
            return

        _,c,merge = self.sess.run([self.opim_func,self.cost,self.merged],feed_dict={self.im_in:data_in, self.im_out:data_out,self.is_training:True})
        self.summ_writer.add_summary(merge,step)

        self.lr *= self.lr_decay

        print(' step',step,'new learnign rate',self.lr,'cost',np.sum(np.sum(c)))

        if self.save_path is not None:
            save_path = self.saver.save(self.sess, self.save_path)
            print("Model saved in file: %s" % save_path)


    def test(self, data_in, expected_out):

        if not self.show_test:
            return

        cv2.imshow('data_in', expected_out[0][:,:,0]*255)
        cv2.waitKey(20)
        input = np.array(data_in)
        input = np.reshape(input, [-1,self.inp_sz,self.inp_sz,self.input_channels])

        out = self.sess.run(tf.nn.softmax(self.model),feed_dict={self.im_in:input,self.is_training:False})

        zeros = np.zeros(shape=[self.inp_sz, self.inp_sz, 1])
        gray_scale_out = np.reshape(out[0], [self.inp_sz, self.inp_sz, self.classes])
        gray_scale_out = np.array(gray_scale_out*255, dtype=np.uint8)
        result = np.concatenate((gray_scale_out,zeros),axis=2)

        expected = np.concatenate((expected_out[0]*255, zeros),axis = 2)
        cv2.imshow('expected ', expected)
        im_result = np.concatenate(( expected, result),axis=1)
        cv2.imshow('result ',im_result)

        cv2.waitKey(100)


