import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
from utils import setup, str2list, str2dict
from functools import partial

from data import build_dataloaders
from model import build_model
from losses import Loss
from optims import build_optimizer
from evaluation import Evaluator
from engine import Engine

from torch.utils import tensorboard
from torch.nn import DataParallel

from tabulate import tabulate

from timm.utils import ModelEma
import torch
from thop import profile, clever_format

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # system
    parser.add_argument('--gpus', type=str2list, default='0', help='gpu ids (default: 0)')
    parser.add_argument('--exp', type=str, default='MFENet-SYSU-all_tricks', help='experiment name')
    parser.add_argument('--seed', type=int, default=777, help='random seed (default: 777)')
    parser.add_argument('--config', type=str, default='', help='path to config file')
    
    # data 
    parser.add_argument('--dataroot', type=str, default='/root/data/DataSets', help='path to dataset')
    parser.add_argument('--dataset', type=str, default='sysu', help='dataset name (default: regdb, sysu, llcm)')
    parser.add_argument('--img_size', type=partial(str2list, f=int), default='384,192')
    parser.add_argument('--pixel_mean', type=partial(str2list, f=float), default='0.485,0.456,0.406', help='mean of dataset (default: 0.485,0.456,0.406)')
    parser.add_argument('--pixel_std', type=partial(str2list, f=float), default='0.229,0.224,0.225', help='std of dataset (default: 0.229,0.224,0.225)')
    parser.add_argument('--trn_bs', type=partial(str2list, f=int), default='8,8', help='batch size (default: 64)')
    parser.add_argument('--tst_bs', type=int, default=128, help='batch size (default: 64)')
    parser.add_argument('--num_workers', type=int, default=6, help='number of workers (default: 8)')
    parser.add_argument('--split_num', type=str, default='4', help='')
    parser.add_argument('--pad_tf', type=str2dict, default='do_pad(bool)=1|padding_size(int)=[10,10]|padding_mode(str)=constant|padding_fill(int)=0')
    parser.add_argument('--flip_tf', type=str2dict, default='do_flip(bool)=1|flip_prob(float)=0.5')
    parser.add_argument('--color_tf', type=str2dict, default='do_color_jitter(bool)=0|color_jitter_prob(float)=0.5|brightness(float)=0.0|contrast(float)=0.0|saturation(float)=0.0|hue(float)=0.2')
    parser.add_argument('--gray_tf', type=str2dict, default='do_gray(bool)=0|gray_prob(float)=0.5')
    parser.add_argument('--vimc_tf', type=str2dict, default='do_vimc(bool)=0')
    parser.add_argument('--rwa_tf', type=str2dict, default='do_rwa(bool)=1|rwa_prob(float)=0.5')
    parser.add_argument('--cag_tf', type=str2dict, default='do_cag(bool)=0|cag_prob(float)=0.5')
    parser.add_argument('--mfb_rwa_tf', type=str2dict, default='do_mfb_rwa(bool)=0|mfb_rwa_prob(float)=0.5|mfb_rwa_N(int)=5')
    parser.add_argument('--norm_tf', type=bool, default=True, help='do normalization (default: True)')
    parser.add_argument('--rea_tf', type=str2dict, default='do_rea(bool)=1|rea_prob(float)=0.5|rea_value(float)=0.0|rea_scale(float)=[0.02,0.4]|rea_ratio(float)=[0.3,3.33]')
    parser.add_argument('--crea_tf', type=str2dict, default='do_crea(bool)=0|crea_prob(float)=0.5')

    # testing
    parser.add_argument('--eval_freq', type=partial(str2list, f=int), default='5,100,1', help='test epochs (default: 20)')
    parser.add_argument('--show_nums', type=int, default=50)
    parser.add_argument('--dist_metric', type=str, default='cosine', help='distance metric for re-ranking (default: euclidean)')
    parser.add_argument('--use_cython', type=bool, default=True, help='using cython for evaluation (default: True)')
    parser.add_argument('--search_option', type=int, default=3, help='')
    parser.add_argument('--test_flip', type=bool, default=True, help='')
    
    # model
    parser.add_argument('--model', type=str, default='MFENet', help='model name (default: BoT)')
    parser.add_argument('--model_path', type=str, default='', help='path to pre-trained model (default: None)') #keep_rates(float)=[0.75,0.75,0.75]|keep_list(str)=[6,12,18]|            |keep_rates(int)=[2,2,2,2,2,2,2]|keep_list(str)=[3,6,9,12,15,18,21]|ordered(bool)=1|sample_wise(bool)=1
    parser.add_argument('--model_kwargs', type=str2dict, default='B(int)=7|N(int)=3', help='kwargs for model ("cls_bias(bool)=1|cls_weight(float)=1.0|tst_list(int)=[1,2,3,4]")')
    parser.add_argument('--ema', type=str2dict, default='ema_model(bool)=0|ema_decay(float)=0.9992', help='ema for model (default: ema_model(bool)=1|ema_decay(float)=0.9999)')
    
    # loss
    parser.add_argument('--loss', type=str2list, default='id_loss,wrt_loss,cmsr_loss,cmsr_loss,cmrr_loss,cmrr_loss', help='loss function name (default: CrossEntropyLoss)')
    parser.add_argument('--loss_weights', type=partial(str2list, f=float), default='3.0,3.0,0.3,0.3,0.2,0.02', help='loss weights (default: 1.0 for all losses, can only write one value)')
    parser.add_argument('--loss_nums', type=partial(str2list, f=int), default='3,3,3,3,1,1', help='loss numbers (default: 1 for all losses, can only write one value)')
    parser.add_argument('--loss_kwargs', type=partial(str2list, f=str2dict), default='label_smoothing(float)=0.0,,m(float)=0.1,m(float)=0.1,r(float)=2.0,r(float)=2.0', help='kwargs for loss function ("****")')
    
    # optimizer
    # sysu
    parser.add_argument('--optim', type=str, default='adam', help='optimizer name')
    parser.add_argument('--optim_kwargs', type=str2dict, default='lr(float)=3.5e-4|weight_decay(float)=5e-4', help='kwargs for optimizer ("****")')
    parser.add_argument('--lr_scheduler', type=str, default='LinearWarmupLrScheduler')
    parser.add_argument('--lr_scheduler_kwargs', type=str2dict, default='warmup_epochs(int)=9|lr_multiplier(float)=1e-2|lrs2(str)=MultiStepLR|lrs2_kwargs(str2dict)={milestones(int)=[71,111]}')
    parser.add_argument('--epochs', type=int, default=140)
    # regdb
    # parser.add_argument('--optim', type=str, default='adam', help='optimizer name')
    # parser.add_argument('--optim_kwargs', type=str2dict, default='lr(float)=3.5e-4|weight_decay(float)=5e-4', help='kwargs for optimizer ("****")')
    # parser.add_argument('--lr_scheduler', type=str, default='LinearWarmupLrScheduler')
    # parser.add_argument('--lr_scheduler_kwargs', type=str2dict, default='warmup_epochs(int)=0|lr_multiplier(float)=1e-2|lrs2(str)=MultiStepLR|lrs2_kwargs(str2dict)={milestones(int)=[251]}')
    # parser.add_argument('--epochs', type=int, default=300)
    
    parser.add_argument('--freeze_bb', type=int, default=0)
    parser.add_argument('--eval_bb', type=int, default=0)
    parser.add_argument('--amp', type=bool, default=True, help='using amp for training (default: False)')
    parser.add_argument('--grad_clip', type=float, default=0.0, help='gradnorm clip (default: 0)') # msmt17: 30.0; Market1501: 25.0; dukemtmc: 30.0; CUHK03: 25.0
    
    args = parser.parse_args()
    
    tb_writer, logger, savedir = setup(args, determenistic=True, benchmark=False)
    
    trn_dl, gal_dl, que_dl = build_dataloaders(args)

    mdl = build_model(args, trn_dl).to('cuda')
    
    mdl.eval()
    tensor = (torch.rand(1, 3, *args.img_size).cuda(), torch.Tensor([2]).long().cuda())
    macs, params = profile(mdl, inputs=tensor)
    macs, params = clever_format([macs, params], "%.3f")
    logger.info(f"\tMACs: {macs}, Params: {params}")

    if args.ema['ema_model']:
        mdl_ema = ModelEma(mdl, decay=args.ema['ema_decay'])
        logger.info(f'\tUsing EMA with decay {args.ema["ema_decay"]}')
    else:
        mdl_ema = None

    optim, lrs = build_optimizer(args, mdl)
    certifier = Loss(args).to('cuda')
    eva = Evaluator(args, mdl, [que_dl, gal_dl], tb_writer)
    eng = Engine(args, mdl, mdl_ema, optim, lrs, certifier, eva, trn_dl, tb_writer, savedir)
    
    eng.train_test()
    
    
    #  test mode
    # import torch
    # mdl.load_state_dict(torch.load('/root/data/Fast_VIReID/runs/2025-03-02_21:11:51-------Baseline2-256x128-minsampler-spatial-new/best_ckpt.pth')['model'])
    # eva(-1)
    
    
    

        
    
    
    
    
    
    
    
    