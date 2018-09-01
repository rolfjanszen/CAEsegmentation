import cv2
import numpy as np
import pickle as pkl

import os

class imageGetter:

    filelist =[]
    input_path = ''
    output_path= ''

    def __init__(self, inputpath, outpputpath):
        self.filelist = [file for file in os.listdir(inputpath) if file.endswith('.png')]
        self.outpput_path = outpputpath
        self.input_path = inputpath

    def getImageSubset(self,begin, end):
        image_sublist = self.filelist[begin: end]

        images_inp =[]
        images_outp = []

        for image_name in image_sublist:

            im_in_path = os.path.join(self.input_path, image_name)
            im = cv2.imread(im_in_path)
            images_inp.append(im)
            im_out_path = os.path.join(self.outpput_path, image_name)

            im = cv2.imread(im_out_path)
            images_outp.append(im)

        return images_inp, images_outp
