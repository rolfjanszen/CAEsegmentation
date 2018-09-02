from pre_process import imageGetter
from cae import CAE
inp_path = '../data/train/images'
out_path = '../data/train/masks'
save_path = '../model/salt_segmentation.ckpt'

imget = imageGetter(inp_path, out_path)
autoEnc = CAE(101,101, save_path)

batch_size = 60
range_start = 0
epochs = 100

for i in range(epochs):
    for range_end in range(batch_size,len(imget.filelist), batch_size):
        images_inp, images_outp = imget.getImageSubset(range_start, range_end)
        range_start = range_end
        autoEnc.train(images_inp, images_outp)