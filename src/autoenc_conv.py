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

inp_path =path_home+'/data/train/images'
out_path =path_home+'/data/train/masks'
save_path =path_home+'/output/salt_segmentation_test.ckpt'
save_path =None
use_sync_data = False

if use_sync_data:
    x_sz = y_sz = 101
else:
    x_sz = y_sz = 101

imget = imageGetter(inp_path, out_path, x_sz , y_sz)
autoEnc = CAE( x_sz , y_sz, save_path, on_home)

batch_size = 60
range_start = 0
epochs = 100
#
# for i in range(100):
#     images_inp, images_outp = imget.getImageSubset(999+i, 1000+i)
#     autoEnc.test(images_inp[0], images_outp[0])

for i in range(epochs):
    print('running epoch ',i)
    for range_end in range(batch_size,len(imget.filelist), batch_size):
        # TODO get random image set
        images_inp, images_outp = imget.getImageSubset(range_start, range_end,use_sync_data)
        # images_inp, images_outp = imget.create_test_data(range_start, range_end)

        autoEnc.train(images_inp, images_outp)
        range_start = range_end

