import cv2
import numpy as np
import pickle as pkl
import time
import os

class imageGetter:

    filelist =[]
    input_path = ''
    output_path= ''

    im_in_dimension =[]
    im_out_dimension = []
    stripe_im =[]
    def __init__(self, inputpath, outpputpath, test_x_, test_y_):


        self.filelist = [file for file in os.listdir(inputpath) if file.endswith('.png')]
        self.outpput_path = outpputpath
        self.input_path = inputpath
        self.stripe_im = self.create_stripe_pattern(test_x_, test_y_)
        self.test_x = test_x_
        self.test_y = test_y_

    def getImageSubset(self,begin, end, create_test = False):

        image_sublist = self.filelist[begin: end]

        images_inp =[]
        images_outp = []
        x_sz = y_sz = 50
        t = time.time()
        for image_name in image_sublist:
            im_out_path = os.path.join(self.outpput_path, image_name)


            im = cv2.imread(im_out_path,cv2.IMREAD_GRAYSCALE)
            im = cv2.resize(im, (self.test_x, self.test_y))
            # cv2.imshow('out', im)
            # cv2.waitKey()
            _, im = cv2.threshold(im, 19, 1, cv2.THRESH_BINARY)

            salt_area = np.sum(np.sum(im))/(self.test_x)+10
            rand_area = np.random.randint(0,self.test_y)

            if salt_area > 40 or rand_area < salt_area:

                # cv2.imshow('pos out', im*250)
                neg_im = 1-im
                im = np.reshape(im, [im.shape[0], im.shape[1], 1])
                neg_im = np.reshape(neg_im, [im.shape[0], im.shape[1], 1])
                class_im = np.concatenate((im,neg_im),axis =2)
                images_outp.append(class_im)

                if create_test:
                    im_in = cv2.imread(im_out_path, cv2.IMREAD_GRAYSCALE)
                    _, im = cv2.threshold(im_in, 19, 1, cv2.THRESH_BINARY)
                    res_im = cv2.resize(im, (self.test_x,self.test_y))
                    im_in = self.stripe_im * res_im
                    # cv2.imshow('test inp', im_in)
                else:
                    im_in_path = os.path.join(self.input_path, image_name)
                    im_in = cv2.imread(im_in_path,cv2.IMREAD_GRAYSCALE)
                    im_in = cv2.resize(im, (self.test_x, self.test_y))
                images_inp.append(im_in)

            # cv2.imshow('neg out', neg_im*250)
            # cv2.waitKey()

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









