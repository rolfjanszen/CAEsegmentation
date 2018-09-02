from pre_process import imageGetter
from cae import CAE
inp_path = '/home/rj/Downloads/all/train/images'
out_path = '/home/rj/Downloads/all/train/masks'

imget = imageGetter(inp_path, out_path)
autoEnc = CAE(101,101)

batch_size = 60
range_start = 0

for range_end in range(batch_size,len(imget.filelist), batch_size):
    images_inp, images_outp = imget.getImageSubset(range_start, range_end)
    range_start = range_end
    autoEnc.train(images_inp, images_outp)