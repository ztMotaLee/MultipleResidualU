import time
import torch
import random

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image, make_grid
from torch import nn

from utils import *
from options import TrainOptions
from models import PUNet
from losses import LossL1, LossFreqReco, LossSSIM, LossTV
from datasets import PairedImgDataset

print('---------------------------------------- step 1/5 : parameters preparing... ----------------------------------------')
opt = TrainOptions().parse()

gpu_num = torch.cuda.device_count()

set_random_seed(opt.seed)

models_dir, log_dir, train_images_dir, val_images_dir = prepare_dir(opt.results_dir, opt.experiment, delete=(not opt.resume))

writer = SummaryWriter(log_dir=log_dir)

print('---------------------------------------- step 2/5 : data loading... ------------------------------------------------')
print('training data loading...')
train_dataset = PairedImgDataset(data_source=opt.data_source, mode='train', crop=[opt.cropx, opt.cropy], random_resize=None)
train_dataloader = DataLoader(train_dataset, batch_size=opt.train_bs_per_gpu*gpu_num, shuffle=True, num_workers=opt.num_workers, pin_memory=True)
print('successfully loading training pairs. =====> qty:{} bs:{}'.format(len(train_dataset),opt.train_bs_per_gpu*gpu_num))

print('validating data loading...')
val_dataset = PairedImgDataset(data_source=opt.data_source, mode='val')
val_dataloader = DataLoader(val_dataset, batch_size=opt.val_bs, shuffle=False, num_workers=opt.num_workers, pin_memory=True)
print('successfully loading validating pairs. =====> qty:{} bs:{}'.format(len(val_dataset),opt.val_bs))

print('---------------------------------------- step 3/5 : model defining... ----------------------------------------------')
model = nn.DataParallel(PUNet()).cuda()
print_para_num(model)

if opt.pretrained is not None:
    model.load_state_dict(torch.load(opt.pretrained))
    print('successfully loading pretrained model.')
    
print('---------------------------------------- step 4/5 : requisites defining... -----------------------------------------')
criterion_l1 = LossL1().cuda()
criterion_tv = LossTV().cuda()
criterion_fre = LossFreqReco().cuda()

if opt.resume:
    state = torch.load(models_dir + '/latest.pth')
    optimizer = state['optimizer']
    scheduler = state['scheduler']
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, opt.scheduler, 0.5)

print('---------------------------------------- step 5/5 : training... ----------------------------------------------------')
def main():
    
    optimal = [0., 0.]
    start_epoch = 1
    if opt.resume:
        state = torch.load(models_dir + '/latest.pth')
        model.load_state_dict(state['model'])
        start_epoch = state['epoch'] + 1
        optimal = state['optimal']
        
        print('Resume from epoch %d' % (start_epoch), optimal)
    
    for epoch in range(start_epoch, opt.n_epochs + 1):
        train(epoch, optimal)
        
        if (epoch) % opt.val_gap == 0:
            val(epoch, optimal)
        
    writer.close()
    
def train(epoch, optimal):
    model.train()
    
    max_iter = len(train_dataloader)
        
    iter_ssim_meter = AverageMeter()
    iter_timer = Timer()
    
    for i, (imgs_l, imgs_r, gts_l, gts_r) in enumerate(train_dataloader):
        [imgs_l, imgs_r, gts_l, gts_r] = [x.cuda() for x in [imgs_l, imgs_r, gts_l, gts_r]]
        cur_batch = imgs_l.shape[0]
        
        optimizer.zero_grad()
        input = torch.cat([imgs_l, imgs_r], 1)
        preds_l, preds_r = model(input)

        loss_l1 = criterion_l1(preds_l, gts_l) + criterion_l1(preds_r, gts_r)
        loss_tv = criterion_tv(preds_l) + criterion_tv(preds_r)
        loss_fre = criterion_fre(preds_l, gts_l) + criterion_fre(preds_r, gts_r)

        loss = loss_l1 + 0.1*loss_tv + 0.1*loss_fre
        
        loss.backward()
        optimizer.step()
        
        iter_ssim_meter.update(loss.item()*cur_batch, cur_batch)
        
        # if i == 0:
        #     save_image(torch.cat((imgs,preds.detach(),gts),0), train_images_dir + '/epoch_{:0>4}_iter_{:0>4}.png'.format(epoch, i+1), nrow=opt.train_bs_per_gpu, normalize=True, scale_each=True)
            
        if (i+1) % opt.print_gap == 0:
            print('Training: Epoch[{:0>4}/{:0>4}] Iteration[{:0>4}/{:0>4}] Best: {:.4f}/{:.4f} loss: {:.4f} Time: {:.4f} LR: {:.8f}'.format(epoch, 
            opt.n_epochs, i + 1, max_iter, optimal[0], optimal[1], iter_ssim_meter.average(), iter_timer.timeit(), scheduler.get_last_lr()[0]))
            writer.add_scalar('Loss_cont', iter_ssim_meter.average(auto_reset=True), i+1 + (epoch - 1) * max_iter)
            
            
    writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch)
    torch.save({'model': model.state_dict(), 'epoch': epoch, 'optimal': optimal, 'optimizer': optimizer, 'scheduler': scheduler}, models_dir + '/latest.pth')
    scheduler.step()
    
def val(epoch, optimal):
    model.eval()
    
    print(''); print('Validating...', end=' ')
    
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    timer = Timer()
    
    for i, (imgs_l, imgs_r, gts_l, gts_r) in enumerate(val_dataloader):
        [imgs_l, imgs_r, gts_l, gts_r] = [x.cuda() for x in [imgs_l, imgs_r, gts_l, gts_r]]
        h, w = gts_l.size(2), gts_l.size(3)
        [imgs_l, imgs_r] = [check_padding(x) for x in [imgs_l, imgs_r]]
        input = torch.cat([imgs_l, imgs_r], 1)

        with torch.no_grad():
            preds_l, preds_r = model(input)
        # [preds_l, preds_r] = [torch.clamp(x, 0, 1) for x in [preds_l, preds_r]]
        [preds_l, preds_r] = [x[:, :, :h, :w] for x in [preds_l, preds_r]]

        psnr_value, ssim_value = get_metrics(preds_l, gts_l, psnr_only=False)

        psnr_meter.update(psnr_value, imgs_l.shape[0])
        ssim_meter.update(ssim_value, imgs_l.shape[0])
        
        # psnr_meter.update(get_metrics(preds_clip, gts), imgs.shape[0])
        
        # if i == 0:
        #     if epoch == opt.val_gap:
        #         save_image(imgs, val_images_dir + '/epoch_{:0>4}_iter_{:0>4}_img.png'.format(epoch, i+1), nrow=opt.val_bs, normalize=True, scale_each=True)
        #         save_image(gts, val_images_dir + '/epoch_{:0>4}_iter_{:0>4}_gt.png'.format(epoch, i+1), nrow=opt.val_bs, normalize=True, scale_each=True)
        #     save_image(preds_clip, val_images_dir + '/epoch_{:0>4}_iter_{:0>4}_restored.png'.format(epoch, i+1), nrow=opt.val_bs, normalize=True, scale_each=True)
    
    if optimal[0] < psnr_meter.average():
        optimal[0] = psnr_meter.average()
        torch.save(model.state_dict(), models_dir + '/optimal_psnr.pth')
        # torch.save(model.state_dict(), models_dir + '/optimal_{:.2f}_epoch_{:0>4}.pth'.format(optimal[0], epoch))
    if optimal[1] < ssim_meter.average():
        optimal[1] = ssim_meter.average()
        torch.save(model.state_dict(), models_dir + '/optimal_ssim.pth')
        
    writer.add_scalar('psnr', psnr_meter.average(), epoch)
    writer.add_scalar('ssim', ssim_meter.average(), epoch)

    print('Epoch[{:0>4}/{:0>4}] PSNR/SSIM: {:.4f}/{:.4f} Best: {:.4f}/{:.4f} Time: {:.4f}'.format(epoch, opt.n_epochs, psnr_meter.average(),
     ssim_meter.average(), optimal[0], optimal[1],timer.timeit())); print('')
    
    # torch.save(model.state_dict(), models_dir + '/epoch_{:0>4}.pth'.format(epoch))
    
if __name__ == '__main__':
    main()
    