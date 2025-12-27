import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0' 
import torch
print(torch.cuda.is_available())

import matplotlib.pyplot as plt
import argparse
import builtins
import math
import torch.nn.functional as F
import pywt

import random
import shutil
import time
import warnings
import sys

sys.path.insert(0, "./")

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
from denoiser.config import Config
from denoiser.modeling.architectures import build_architecture
from denoiser.solver import make_lr_scheduler, make_optimizer
from denoiser.data.bulid_data import build_dataset
from denoiser.utils.miscellaneous import mkdir, save_config
import numpy as np
from pytorch_wavelets import DWTForward, DWTInverse
from AFM_B import AFM_B

from imageio import imwrite
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import interpolate

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
parser.add_argument(
    "--config_file",
    # default="config.py",
    default="./seg512_factgus_1111_l1.py",
    metavar="FILE",
    help="path to config file",
    type=str,
)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, activation, mid_channels=None, use_bn=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if use_bn:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(mid_channels),
                activation,
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                activation
            )
        else:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                activation,
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                activation
            )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, activation, use_bn=False):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, activation, use_bn=use_bn)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, activation, bilinear=False, use_bn=False):
        super().__init__()

        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, activation, in_channels // 2, use_bn=use_bn)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, activation, use_bn=use_bn)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    

class UNet3(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, residual=False, activation_type="relu", use_bn=True):
        super(UNet3, self).__init__()

        if activation_type == "leaky_relu":
            activation = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        elif activation_type == "relu":
            activation = nn.ReLU(inplace=False)
        else:
            raise TypeError

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 96, activation, use_bn=use_bn)
        self.down1 = Down(96, 96*2, activation, use_bn=use_bn)
        self.down2 = Down(96*2, 96*4, activation, use_bn=use_bn)
        self.down3 = Down(96*4, 96*8, activation, use_bn=use_bn)
        
        self.up1 = Up(96*8, 96*4, activation, use_bn=use_bn)
        self.up2 = Up(96*4, 96*2, activation, use_bn=use_bn)
        self.up3 = Up(96*2, 96*1, activation, use_bn=use_bn)
        self.outc = OutConv(96, n_classes)
        self.residual = residual

    def forward(self, input):
        x1 = self.inc(input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)
        if self.residual:
            x = input + x
        return x

class UNet4(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, residual=False, activation_type="relu", use_bn=True):
        super(UNet4, self).__init__()

        if activation_type == "leaky_relu":
            activation = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        elif activation_type == "relu":
            activation = nn.ReLU(inplace=False)
        else:
            raise TypeError

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 96, activation, use_bn=use_bn)
        self.down1 = Down(96, 96*2, activation, use_bn=use_bn)
        self.down2 = Down(96*2, 96*4, activation, use_bn=use_bn)
        self.down3 = Down(96*4, 96*8, activation, use_bn=use_bn)
        
        self.up1 = Up(96*8, 96*4, activation, use_bn=use_bn)
        self.up2 = Up(96*4, 96*2, activation, use_bn=use_bn)
        self.up3 = Up(96*2, 96*1, activation, use_bn=use_bn)
        self.outc = OutConv(96, n_classes)
        self.residual = residual

    def forward(self, input):
        x1 = self.inc(input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)
        if self.residual:
            x = input + x
        return x
    
class UNet5(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, residual=False, activation_type="relu", use_bn=True):
        super(UNet5, self).__init__()

        if activation_type == "leaky_relu":
            activation = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        elif activation_type == "relu":
            activation = nn.ReLU(inplace=False)
        else:
            raise TypeError

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 96, activation, use_bn=use_bn)
        self.down1 = Down(96, 96*2, activation, use_bn=use_bn)
        self.down2 = Down(96*2, 96*4, activation, use_bn=use_bn)
        self.down3 = Down(96*4, 96*8, activation, use_bn=use_bn)
        
        self.up1 = Up(96*8, 96*4, activation, use_bn=use_bn)
        self.up2 = Up(96*4, 96*2, activation, use_bn=use_bn)
        self.up3 = Up(96*2, 96*1, activation, use_bn=use_bn)
        self.outc = OutConv(96, n_classes)
        self.residual = residual

    def forward(self, input):
        x1 = self.inc(input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)
        if self.residual:
            x = input + x
        return x
    
class UNet6(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, residual=False, activation_type="relu", use_bn=True):
        super(UNet6, self).__init__()

        if activation_type == "leaky_relu":
            activation = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        elif activation_type == "relu":
            activation = nn.ReLU(inplace=False)
        else:
            raise TypeError

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 96, activation, use_bn=use_bn)
        self.down1 = Down(96, 96*2, activation, use_bn=use_bn)
        self.down2 = Down(96*2, 96*4, activation, use_bn=use_bn)
        self.down3 = Down(96*4, 96*8, activation, use_bn=use_bn)
        
        self.up1 = Up(96*8, 96*4, activation, use_bn=use_bn)
        self.up2 = Up(96*4, 96*2, activation, use_bn=use_bn)
        self.up3 = Up(96*2, 96*1, activation, use_bn=use_bn)
        self.outc = OutConv(96, n_classes)
        self.residual = residual

    def forward(self, input):
        x1 = self.inc(input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)
        if self.residual:
            x = input + x
        return x


def clip_to_uint8(arr):
    if isinstance(arr, np.ndarray):
        return np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)
    x = torch.clamp(arr * 255.0 + 0.5, 0, 255).to(torch.uint8)
    return x


def calculate_psnr(a, b, axis=None):
    
    if isinstance(a, np.ndarray):
        a, b = [x.astype(np.float32) for x in [a, b]]
        x = np.mean((a - b) ** 2, axis=axis)
        return np.log10((a.max() * a.max()) / x) * 10.0
    a, b = [x.to(torch.float32) for x in [a, b]]
    x = torch.mean((a - b) ** 2)
    return torch.log((a.max() * a.max()) / x) * (10.0 / math.log(10))

def normalize(data):

    data = np.array(data)
    min_val = np.min(data)
    max_val = np.max(data)
    if min_val == max_val:
        return np.full_like(data, 0.5)  
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data


def main():
    args = parser.parse_args()

    cfg = Config.fromfile(args.config_file)
    output_dir = cfg.results.output_dir
    if output_dir:
        mkdir(output_dir)

    output_config_path = os.path.join(output_dir, "config.py")
    save_config(cfg, output_config_path)

    if cfg.seed is not None:
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    if cfg.gpu is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely "
            "disable data parallelism."
        )

    if cfg.dist_url == "env://" and cfg.world_size == -1:
        cfg.world_size = int(os.environ["WORLD_SIZE"])

    cfg.distributed = cfg.world_size > 1 or cfg.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if cfg.multiprocessing_distributed:

        cfg.world_size = ngpus_per_node * cfg.world_size

        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, cfg.copy()))
    else:

        main_worker(cfg.gpu, ngpus_per_node, cfg)


def main_worker(gpu, ngpus_per_node, cfg):
    cfg.gpu = gpu


    if cfg.multiprocessing_distributed and cfg.gpu != 0:

        def print_pass(*cfg):
            pass

        builtins.print = print_pass

    if cfg.gpu is not None:
        print("Use GPU: {} for training".format(cfg.gpu))

    if cfg.distributed:
        if cfg.dist_url == "env://" and cfg.rank == -1:
            cfg.rank = int(os.environ["RANK"])
        if cfg.multiprocessing_distributed:

            cfg.rank = cfg.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=cfg.dist_backend,
            init_method=cfg.dist_url,
            world_size=cfg.world_size,
            rank=cfg.rank,
        )

    model = build_architecture(cfg.model)
    print(model)
    

    model_DWT_CA = UNet3(n_channels =1,n_classes=1)
    model_DWT_CA =model_DWT_CA.to(cfg.gpu)
    print(model_DWT_CA)

    model_DWT_CV = UNet3(n_channels =1,n_classes=1)
    model_DWT_CV =model_DWT_CV.to(cfg.gpu)
    print(model_DWT_CV)

    model_DWT_CH = UNet3(n_channels =1,n_classes=1)
    model_DWT_CH =model_DWT_CH.to(cfg.gpu)
    print(model_DWT_CH)

    model_DWT_CD = UNet3(n_channels =1,n_classes=1)
    model_DWT_CD =model_DWT_CD.to(cfg.gpu)
    print(model_DWT_CD)



    if cfg.distributed:

        if cfg.gpu is not None:
            torch.cuda.set_device(cfg.gpu)
            model.cuda(cfg.gpu)

            cfg.batch_size = int(cfg.batch_size / ngpus_per_node)
            cfg.workers = int((cfg.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[cfg.gpu]
            )
        else:
            model.cuda()
            
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif cfg.gpu is not None:
        torch.cuda.set_device(cfg.gpu)
        model = model.cuda(cfg.gpu)
        
    else:
        
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    optimizer = make_optimizer(cfg, model)

    optimizer_DWT_CA = make_optimizer(cfg, model_DWT_CA)
    optimizer_DWT_CV = make_optimizer(cfg, model_DWT_CV)
    optimizer_DWT_CD = make_optimizer(cfg, model_DWT_CD)
    optimizer_DWT_CH = make_optimizer(cfg, model_DWT_CH)




    if "lr_type" in cfg.solver:
        scheduler = make_lr_scheduler(cfg, optimizer)
        scheduler2 = make_lr_scheduler(cfg, optimizer_DWT_CA)
        scheduler3 = make_lr_scheduler(cfg, optimizer_DWT_CD)
        scheduler4 = make_lr_scheduler(cfg, optimizer_DWT_CV)
        scheduler5 = make_lr_scheduler(cfg, optimizer_DWT_CH)
    
    if cfg.resume:
        if os.path.isfile(cfg.resume):
            print("=> loading checkpoint '{}'".format(cfg.resume))
            if cfg.gpu is None:
                checkpoint = torch.load(cfg.resume)
            else:
               
                loc = "cuda:{}".format(cfg.gpu)
                checkpoint = torch.load(cfg.resume, map_location=loc)
            cfg.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    cfg.resume, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(cfg.resume))

    cudnn.benchmark = True


    train_dataset = build_dataset(cfg.data_train)

    if cfg.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=(train_sampler is None),
        num_workers=cfg.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )

    dataset_val = build_dataset(cfg.data_test)
    val_loader = torch.utils.data.DataLoader(
        dataset_val, batch_size=1, shuffle=False, num_workers=0
    )

    writer = SummaryWriter(log_dir="{}/log_{}".format(cfg.results.output_dir, cfg.rank))
    psnr_best = 0
    best_epoch = 0
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(cfg.start_epoch, cfg.epochs):
        if cfg.distributed:
            train_sampler.set_epoch(epoch)

        train(train_loader, model, model_DWT_CA,model_DWT_CD,model_DWT_CH,model_DWT_CV, optimizer,optimizer_DWT_CA,optimizer_DWT_CD,optimizer_DWT_CH,optimizer_DWT_CV, epoch, cfg, writer)
        
        scheduler.step()
        scheduler2.step()
        scheduler3.step()
        scheduler4.step()
        scheduler5.step()

        if not cfg.multiprocessing_distributed or (
            cfg.multiprocessing_distributed
            and cfg.rank % ngpus_per_node == 0
            and (epoch + 1) % cfg.test_freq == 0
        ):
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "optimizer": optimizer_DWT_CV.state_dict(),
                    "optimizer": optimizer_DWT_CA.state_dict(),
                    "optimizer": optimizer_DWT_CD.state_dict(),
                    "optimizer": optimizer_DWT_CH.state_dict(),
                },
                is_best=False,
                filename="{}/checkpoint_{:04d}.pth.tar".format(
                    cfg.results.output_dir, epoch
                ),
            )
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "optimizer": optimizer_DWT_CV.state_dict(),
                    "optimizer": optimizer_DWT_CA.state_dict(),
                    "optimizer": optimizer_DWT_CD.state_dict(),
                    "optimizer": optimizer_DWT_CH.state_dict(),
                },
                is_best=False,
                filename="{}/checkpoint_last.pth.tar".format(cfg.results.output_dir),
            )
            if (epoch + 1) == cfg.epochs:
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "optimizer": optimizer_DWT_CV.state_dict(),
                        "optimizer": optimizer_DWT_CA.state_dict(),
                        "optimizer": optimizer_DWT_CD.state_dict(),
                        "optimizer": optimizer_DWT_CH.state_dict(),
                    },
                    is_best=False,
                    filename="{}/checkpoint_final.pth.tar".format(
                        cfg.results.output_dir
                    ),
                )
            model.eval()
            model_DWT_CA.eval()      
            model_DWT_CD.eval()
            model_DWT_CH.eval()
            model_DWT_CV.eval()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
            psnrs = [0]
            
            for _, (images, images_clean, std, idx) in enumerate(val_loader):
                inputs = images.to(cfg.gpu)
                
                with torch.no_grad():
                    outputs1 = model(inputs)
                
                    outputs1 =outputs1.cpu()
                    xfm = DWTForward(wave='haar', mode='zero')
                    YL2, YH2 = xfm(outputs1)
                    cA = YL2
                    cH = YH2[0][:, :, 0, :, :] 
                    cV = YH2[0][:, :, 1, :, :]  
                    cD = YH2[0][:, :, 2, :, :]  
                    cA=cA.to(cfg.gpu)
                    cD=cD.to(cfg.gpu)
                    cV=cV.to(cfg.gpu)
                    cH=cH.to(cfg.gpu)
                    

                    cV1 = model_DWT_CV(cV)
                    cV1= cV1.cpu()
                    cA1 = model_DWT_CA(cA)
                    cA1= cA1.cpu()
                    cH1 = model_DWT_CH(cH)
                    cH1= cH1.cpu()
                    cD1 = model_DWT_CD(cD)
                    cD1= cD1.cpu()

                    YH2[0][:, :, 1, :, :] = cV1   
                    YL2=cA1
                    YH2[0][:, :, 0, :, :] = cH1 
                    YH2[0][:, :, 2, :, :] = cD1 
                    ifm = DWTInverse(wave='haar', mode='zero')
                    outputs = ifm((YL2, YH2))
                    outputs =outputs.to(cfg.gpu)
                
                images_clean = images_clean.to(cfg.gpu)

                psnr = calculate_psnr(outputs, images_clean)
                psnrs.append(psnr.cpu().numpy())

                if idx == 0:

                    if len(images.shape) > 4:
                        img_noise = images[0][0].cpu().numpy().transpose([1, 2, 0])
                    else:
                        img_noise = images[0].cpu().numpy().transpose([1, 2, 0])

                    img_pred = outputs[0].detach().cpu().numpy()
                    img_pred_denosing=outputs1.detach().cpu().numpy()
                    img_clean = images_clean[0].cpu().numpy()
                    base_folder = "{}/{}".format(cfg.results.output_dir, epoch)

                    noise = img_noise.reshape(128,128)
                    pred = img_pred.reshape(128,128)

                    pred_denosing = img_pred_denosing.reshape(128,128)
                    clean = img_clean.reshape(128,128)

                    
                    pred_noise = pred - noise
                    np.save("{}_noise.npy".format(base_folder), noise)
                    np.save("{}_pred.npy".format(base_folder), pred)
                    np.save("{}_pred_denosing.npy".format(base_folder), pred_denosing)
                    np.save("{}_clean.npy".format(base_folder), clean)
                    np.save("{}_pred_noise.npy".format(base_folder), pred_noise)
                    
                   


            print(psnrs)
            psnr_mean = np.array(psnrs).mean()
            writer.add_scalar("PSNR/test", psnr_mean, epoch)
            if psnr_best < psnr_mean:
                psnr_best = psnr_mean
                best_epoch = epoch
            print(psnr_mean)
            print("Best PSNR: {}, epoch: {}".format(psnr_best, best_epoch))

        

def train(train_loader, model, model_DWT_CA,model_DWT_CD,model_DWT_CH,model_DWT_CV, optimizer,optimizer_DWT_CA,optimizer_DWT_CD,optimizer_DWT_CH,optimizer_DWT_CV, epoch, cfg, writer):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    correlation_values = []


    losses2 = AverageMeter("Loss2", ":.4e")

    lr = AverageMeter("lr", ":.9f")
    top1 = AverageMeter("Acc@1", ":6.2f")



    lr.update(optimizer.param_groups[0]["lr"])
    lr.update(optimizer_DWT_CA.param_groups[0]["lr"])
    lr.update(optimizer_DWT_CH.param_groups[0]["lr"])
    lr.update(optimizer_DWT_CV.param_groups[0]["lr"])
    lr.update(optimizer_DWT_CD.param_groups[0]["lr"])
    
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, lr],
        prefix="Epoch: [{}]".format(epoch),
    )
    

    model.train()
    model_DWT_CA.train()
    model_DWT_CH.train()
    model_DWT_CV.train()
    model_DWT_CD.train()


    losses_list = []

    end = time.time()
    for i, (images_noise, images_target, std, idx) in enumerate(train_loader):

        data_time.update(time.time() - end)




        if cfg.gpu is not None:

            images_noise = images_noise.cuda(cfg.gpu)
            images_target = images_target.cuda(cfg.gpu)



        batch_size = images_target.size(0)

        if len(images_noise.shape) == 5:
            images_noise = images_noise.reshape(
                [
                    -1,
                    images_noise.shape[2],
                    images_noise.shape[3],
                    images_noise.shape[4],
                ]
            )
            images_target = images_target.reshape(images_noise.shape)
        

        outputs,loss = model(images_noise, target=images_target)

        out = images_target.cpu()
        xfm = DWTForward(wave='haar', mode='zero')
        YL1, YH1 = xfm(out)
        cA1=YL1
        cH1 = YH1[0][:, :, 0, :, :]  
        cV1 = YH1[0][:, :, 1, :, :]  
        cD1 = YH1[0][:, :, 2, :, :]  

        sampled=images_noise.cpu()
        xfm = DWTForward(wave='haar', mode='zero')
        YL2, YH2 = xfm(sampled)
        cA2=YL2
        cH2 = YH2[0][:, :, 0, :, :]  
        cV2 = YH2[0][:, :, 1, :, :]  
        cD2 = YH2[0][:, :, 2, :, :]  



        cV1=cV1.to(cfg.gpu)
        cV2=cV2.to(cfg.gpu)
        cA1=cA1.to(cfg.gpu)
        cA2=cA2.to(cfg.gpu)
        cH1=cH1.to(cfg.gpu)
        cH2=cH2.to(cfg.gpu)
        cD1=cD1.to(cfg.gpu)
        cD2=cD2.to(cfg.gpu)

        cA3 = torch.tensor(cA1, dtype=torch.float32)  
        cH3 = torch.tensor(cH1, dtype=torch.float32)  
        cH4 = torch.tensor(cH2, dtype=torch.float32)
        cV3 = torch.tensor(cV1, dtype=torch.float32)  
        cV4 = torch.tensor(cV2, dtype=torch.float32)
        cD3 = torch.tensor(cD1, dtype=torch.float32) 
        cD4 = torch.tensor(cD2, dtype=torch.float32)

        cA3.requires_grad_(True)
        cA4.requires_grad_(True)
        cH3.requires_grad_(True)
        cH4.requires_grad_(True)
        cV3.requires_grad_(True)
        cV4.requires_grad_(True)
        cD3.requires_grad_(True)
        cD4.requires_grad_(True)


        cV1_UNET = model_DWT_CV(cV4)
        cA1_UNET = model_DWT_CA(cA4)
        cH1_UNET = model_DWT_CH(cH4)
        cD1_UNET = model_DWT_CD(cD4)



        loss_cV= F.mse_loss(cV1_UNET, cV3, reduction='mean')
        loss_cA= F.mse_loss(cA1_UNET, cA3, reduction='mean')
        loss_cH= F.mse_loss(cH1_UNET, cH3, reduction='mean')
        loss_cD= F.mse_loss(cD1_UNET, cD3, reduction='mean')

        optimizer_DWT_CA.zero_grad()

        loss_cA.backward(retain_graph=True)
        optimizer_DWT_CA.step()


        optimizer_DWT_CH.zero_grad()

        loss_cH.backward(retain_graph=True)
        optimizer_DWT_CH.step()

        optimizer_DWT_CD.zero_grad()

        loss_cD.backward(retain_graph=True)
        optimizer_DWT_CD.step()

        optimizer_DWT_CV.zero_grad()

        loss_cV.backward(retain_graph=True)
        optimizer_DWT_CV.step()


        img =(images_noise-images_target)


        tensor1_flat = outputs.view(-1, 1*64*64)
        tensor2_flat = img.view(-1, 1*64*64)


        tensor1_flat = tensor1_flat.detach()
        tensor2_flat = tensor2_flat.detach()

        
        tensor1_flat = tensor1_flat.cpu()
        tensor2_flat = tensor2_flat.cpu()


        array1 = tensor1_flat.numpy()
        array2 = tensor2_flat.numpy()


        corr_matrix = np.corrcoef(array1.flatten(), array2.flatten())


        correlation_coefficient = np.mean(np.triu(corr_matrix, k=1))
        correlation = torch.tensor(correlation_coefficient**2, device=cfg.gpu)
        correlation_values.append(correlation)        
        
        
        
        

       
        outputs = outputs.to(cfg.gpu)
        images_target = images_target.to(cfg.gpu)

        rgb_to_gray = torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1).to(cfg.gpu)
        grayscale_target = torch.sum(images_target * rgb_to_gray, dim=1, keepdim=True)
        grayscale_pred = torch.sum(outputs * rgb_to_gray, dim=1, keepdim=True)

        # 计算像素级 L1 损失
        pixel_wise_loss = F.mse_loss(outputs, images_target, reduction='mean')

        # 计算亮度 L1 损失
        luminance_loss = F.mse_loss(grayscale_pred, grayscale_target, reduction='mean')
        
        # 组合总损失

        loss1 = pixel_wise_loss + luminance_loss

        
        
        loss_denoising = loss1*(1+correlation)       
       
        # loss_denoising = F.mse_loss(outputs,images_target, reduction='mean')

        #cv_unet= model_DWT(cV4)
        lammda=10
        # a=model_DWT(cV4)
        # b=cV3
        loss_all=loss_denoising   # + lammda* F.mse_loss(a, b, reduction='mean')
        # 更新去噪网络
        optimizer.zero_grad()
        # optimizer_DWT.zero_grad()
        #retain_graph=True是为了保留自动微分的中间张量
        loss_all.backward()
        optimizer.step()
        # optimizer_DWT.step()
        # for name, param in model.named_parameters():
        #     print(f"After Epoch {epoch}, {name}: {param.data}")
        # print("Epoch:", epoch)
        # print("BatchNorm weight:", model.base_net.up3.conv.double_conv[1].weight)
        # print("BatchNorm bias:", model.base_net.up3.conv.double_conv[1].bias)

        writer.add_scalar("Loss/ALL", loss_all.item(), epoch * len(train_loader) + i)
        # writer.add_scalar("Loss/CH", loss_cH.item(), epoch * len(train_loader) + i)
        # writer.add_scalar("Loss/CH", loss_cH.item(), epoch * len(train_loader) + i)
        # writer.add_scalar("Loss/CV", loss_cV.item(), epoch * len(train_loader) + i)
        writer.add_scalar("Loss/CV", loss_cV.item(), epoch * len(train_loader) + i)
        writer.add_scalar("Loss/DENOISING", loss_denoising.item(), epoch * len(train_loader) + i)
        # # 记录和显示进度
        batch_time.update(time.time() - end)
        end = time.time()
        if i % cfg.print_freq == 0:
            progress.display(i)

    # 保存损失值
    # np.save('train_loss.npy', np.array(losses_list))





def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum =self.sum+ val * n
        self.count =self.count+ n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries =entries+ [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


if __name__ == "__main__":
    main()
