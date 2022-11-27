import numpy as np
import argparse
import matplotlib.pyplot as plt
import torch
from glob import glob
from ntpath import basename
from scipy.misc import imread,imresize
import cv2
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr
from skimage.color import rgb2gray
# import lpips
# from niqe.niqe import niqe

def parse_args():
    parser = argparse.ArgumentParser(description='script to compute all statistics')
    parser.add_argument('--data-path', help='Path to ground truth data', type=str)
    parser.add_argument('--output-path', help='Path to output data', type=str)
    parser.add_argument('--debug', default=0, help='Debug', type=int)
    args = parser.parse_args()
    return args


def compare_mae(img_true, img_test):
    img_true = img_true.astype(np.float32)
    img_test = img_test.astype(np.float32)
    return np.sum(np.abs(img_true - img_test)) / np.sum(img_true + img_test)

def compare_lpips(img_true, img_test, model):
    img_true = torch.from_numpy(img_true).cuda()
    img_test = torch.from_numpy(img_test).cuda()
    img_true = img_true.permute(2,0,1)
    img_test = img_test.permute(2,0,1)
    
#     loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization
    d = model(img_true, img_test)
    return d.cpu().detach().numpy()

args = parse_args()
for arg in vars(args):
    print('[%s] =' % arg, getattr(args, arg))

path_true = args.data_path
path_pred = args.output_path

# loss_fn_alex = lpips.LPIPS(net='alex').cuda() # best forward scores vgg alex
# loss_fn_vgg = lpips.LPIPS(net='vgg').cuda() # best forward scores vgg alex

psnr = []
ssim = []
mae = []
lpips_value_alex = []
lpips_value_vgg = []
names = []
index = 1

files = list(glob(path_true + '/*/*.jpg')) + list(glob(path_true + '/*/*.png')) + list(glob(path_true + '/*.jpg')) + list(glob(path_true + '/*.png'))

for fn in sorted(files):
    name = basename(str(fn))
    names.append(name)
    
    img_gt = (imread(str(fn)) / 255.0).astype(np.float32)
    img_pred = (imread(path_pred + '/' + basename(str(fn))) / 255.0).astype(np.float32)
    
    psnr.append(compare_psnr(img_gt, img_pred, data_range=1))
    ssim.append(compare_ssim(img_gt, img_pred, data_range=1, multichannel=True))
    mae.append(compare_mae(img_gt, img_pred))
    
#     lpips_value_alex.append(compare_lpips(img_gt, img_pred, loss_fn_alex))
#     lpips_value_vgg.append(compare_lpips(img_gt, img_pred, loss_fn_vgg))
    

print(
    "PSNR: %.4f" % round(np.mean(psnr), 4),
    "SSIM: %.4f" % round(np.mean(ssim), 4),
#     'LPIPS(alex): %.4f' % round(np.mean(lpips_value_alex), 4),
#     'LPIPS(vgg): %.4f' % round(np.mean(lpips_value_vgg), 4),
    "MAE: %.4f" % round(np.mean(mae), 4),
)
