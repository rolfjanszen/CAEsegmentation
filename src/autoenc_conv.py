from pre_process import imageGetter
from cae import CAE
import os

# floyd run --gpu --env tensorflow-1.3 --data rolfj/datasets/oil_salts/1:data 'python3 autoenc_conv.py'

base_path = os.path.abspath('')
on_home = False #or am I on a server?
path_home = ''

if 'rj' in base_path:
    print('on home computer')
    on_home =True
    path_home ='..'

#Salt
inp_path  = path_home+'/data/train/images'
out_path  = path_home+'/data/train/masks'
save_path = path_home+'/output/salt_segmentation_test.ckpt'
save_path = path_home+'/output'
depth_csv = path_home +'/data/depths.csv'
file_end_inp = False
file_end_outp = False
input_channels = 2
use_tresh_hold = True

#Carvana
inp_path = path_home + '/data/train_car'
out_path = path_home + '/data/train_masks_car'
save_path = path_home + '/output/carvana.ckpt'
save_path = path_home + '/output_car'
depth_csv = False
file_end_inp = '.jpg'
file_end_outp = '_mask.gif'
input_channels = 1
use_tresh_hold = False
# save_path =None
use_sync_data = False

if use_sync_data:

    x_sz = y_sz = 64
else:
    x_sz = y_sz = 128

imget = imageGetter(inp_path, depth_csv, out_path, x_sz , y_sz,file_end_inp)
autoEnc = CAE( x_sz , y_sz, save_path, on_home,input_channels)

batch_size = 50
range_start = 0
epochs = 100
#
# for i in range(100):
#     images_inp, images_outp = imget.getImageSubset(999+i, 1000+i)
#     autoEnc.test(images_inp[0], images_outp[0])
iteration =0
for i in range(epochs):
    print('running epoch ',i)
    for range_end in range(batch_size,len(imget.filelist), batch_size):
        # TODO get random image set
        images_inp, images_outp = imget.getImageSubset(range_start, batch_size, use_sync_data,file_end_outp)
        # images_inp, images_outp = imget.create_test_data(range_start, range_end)

        autoEnc.train(images_inp, images_outp,iteration)
        test_images_inp, test_images_outp = imget.getImageSubset(0, 50, use_sync_data, file_end_outp,use_tresh_hold,'test')
        autoEnc.test(test_images_inp, test_images_outp)
        range_start = range_end
        iteration +=1
