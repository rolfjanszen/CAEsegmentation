import cv2
import numpy as np
import pickle as pkl
import time
import os
import csv

import pandas as pd
from PIL import Image
from random import shuffle
class imageGetter:

    filelist =[]
    input_path = ''
    output_path= ''

    im_in_dimension =[]
    im_out_dimension = []
    stripe_im =[]
    depth_data = False



    def __init__(self, inputpath, depth_input, outpputpath, test_x_, test_y_,file_end = '.png', test_sz = 100, validate_sz = 200):

        # depth_list =[]
        # with open(depth_input, 'rt') as csvfile:
        #     csv_depths = csv.reader(csvfile, delimiter=' ', quotechar='|')
        #     for depth in csv_depths:
        #         depth_list = depth

        if depth_input:
            data = pd.read_csv(depth_input,index_col=0)
            self.depth_data = data.to_dict()['z']

        self.filelist = [file for file in os.listdir(inputpath) if file.endswith(file_end)]

        shuffle(self.filelist)

        self.test_data = self.filelist[-test_sz:]
        self.validate_data = self.filelist[-(validate_sz +test_sz):-test_sz]
        self.filelist= self.filelist[:-(validate_sz +test_sz)]

        self.outpput_path = outpputpath
        self.input_path = inputpath
        self.stripe_im = self.create_stripe_pattern(test_x_, test_y_)
        self.test_x = test_x_
        self.test_y = test_y_




    def get_input_image(self,create_test,im_out_path,image_name):

        if create_test:

            # Create easy to segment data as sanity check.
            im_in = cv2.imread(im_out_path, cv2.IMREAD_GRAYSCALE)
            _, im = cv2.threshold(im_in, 19, 1, cv2.THRESH_BINARY)
            res_im = cv2.resize(im, (self.test_x, self.test_y))
            im_in = self.stripe_im * res_im
            # cv2.imshow('test inp', im_in)
        else:
            index = image_name[:-4]
            if self.depth_data:
                depth = self.depth_data[index]
                depth_channel = np.array([[[depth] * self.test_x] * self.test_y], dtype=np.uint8)

            im_in_path = os.path.join(self.input_path, image_name)
            im_in = cv2.imread(im_in_path, cv2.IMREAD_GRAYSCALE)
            im_in = cv2.resize(im_in, (self.test_x, self.test_y))

        im_in = im_in.reshape([self.test_x, self.test_y, 1])

        if self.depth_data:
            depth_channel = depth_channel.reshape([self.test_x, self.test_y, 1])
            im_in = np.concatenate((im_in, depth_channel), axis=2)

        return im_in
    def getImageSubset(self,begin, batch_size, create_test = False, file_end = False, use_tresh_hold = False, set='train'):
        image_sublist=[]
        if set == 'train':
            image_sublist = self.filelist[begin: ]
        elif set == 'test':
            if batch_size > len(self.test_data):
                batch_size = len(self.test_data)
            image_sublist = self.test_data[begin:batch_size]

        images_inp =[]
        images_outp = []
        x_sz = y_sz = 50
        t = time.time()
        depth_channel = np.array([[300] * self.test_x] * self.test_y,dtype=np.uint8)

        for image_name in image_sublist:
            im_out_path = os.path.join(self.outpput_path, image_name)

            if file_end:
                im_out_path = im_out_path[:-4]
                im_out_path += file_end
            im = np.array(Image.open(im_out_path))

            im = cv2.resize(im, (self.test_x, self.test_y))

            if use_tresh_hold:
                _, im = cv2.threshold(im, 19, 1, cv2.THRESH_BINARY)

            salt_area = np.sum(np.sum(im))/(self.test_x)+10
            rand_area = np.random.randint(0,self.test_y)

            if salt_area > 10 or rand_area < salt_area:

                # cv2.imshow('pos out', im*250)
                neg_im = 1-im
                im = np.reshape(im, [im.shape[0], im.shape[1], 1])
                neg_im = np.reshape(neg_im, [im.shape[0], im.shape[1], 1])
                class_im = np.concatenate((im,neg_im),axis =2)
                images_outp.append(class_im)

                im_in = self.get_input_image(create_test, im_out_path, image_name)

                images_inp.append(np.copy(im_in))

                if len(images_inp) >batch_size:
                    break

        elapsed = time.time()
        print(' time to create data ', elapsed - t)
        return images_inp, images_outp

    def create_stripe_pattern(self, x_sz, y_sz):
        stripe_im = np.array([[0]*x_sz]*y_sz,dtype=np.uint8)

        for x in range(x_sz):
            for y in range(y_sz):
                stripe_im[x,y] = 5*(x%10)

        # cv2.imshow('stripes ', stripe_im)
        # cv2.waitKey()
        return  stripe_im









