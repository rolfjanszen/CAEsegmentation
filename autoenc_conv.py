from pre_process import imageGetter

inp_path = '/home/rj/Downloads/all/train/images'
out_path = '/home/rj/Downloads/all/train/masks'

imget = imageGetter(inp_path, out_path)

imget.getImageSubset(0,100)