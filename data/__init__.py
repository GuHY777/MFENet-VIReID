import os

import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from torch.utils.data import DataLoader
from data.dataset import SYSUDataset, RegDBDataset, LLCMDataset

from data.sampler import EvenlyRandomIdentitySampler, RandomIdentitySampler
from data.data_augmentation import RandomGrayscale
from data.channelaug import ChannelAdapGray, ChannelRandomErasing
from data.freqaugs import RandomWeightedAverage, MFBRandomWeightedAverage
from data.vimc_augs import WeightedGrayscale, ChannelCutMix, SpectrumJitter

from tabulate import tabulate
import logging

logger = logging.getLogger(__name__)


def load_transforms(is_train, mode='', mean_=(0.485, 0.456, 0.406), std_=(0.229, 0.224, 0.225), size_train=(256, 128), size_test=(256, 128),
                    do_pad=True, padding_size=(10, 10), padding_mode='constant', padding_fill=(0, 0, 0),
                    do_flip=True, flip_prob=0.5,
                    do_color_jitter=True, color_jitter_prob=0.5, brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3,
                    do_gray=False, gray_prob=0.5,
                    do_vimc=False,
                    do_rwa=False, rwa_prob=0.5,
                    do_norm=True,
                    do_rea=True, rea_prob=0.5, rea_value=0.0, rea_scale=(0.02, 0.4), rea_ratio=(0.3, 3.33),
                    do_crea=False, crea_prob=0.5,
                    do_cag=False, cag_prob=0.5,
                    do_mfb_rwa=False, mfb_rwa_prob=0.5, mfb_rwa_N=5
                    ):
    res = []

    if is_train:
        logger.info(f"{mode} training transform:")

        res.append(T.Resize(size=size_train, 
                            interpolation=InterpolationMode.BICUBIC))
        logger.info("\tResize: size={}".format(size_train))
            
        if do_pad:
            res.append(T.RandomCrop(size=size_train, 
                                    padding=padding_size, 
                                    pad_if_needed=True, 
                                    fill=padding_fill, 
                                    padding_mode=padding_mode))
            logger.info("\tPadding: size={}, mode={}, fill={}".format(padding_size, padding_mode, padding_fill))
            
        if do_flip:
            res.append(T.RandomHorizontalFlip(p=flip_prob))
            logger.info("\tRandomHorizontalFlip: prob={}".format(flip_prob))
            
        if do_color_jitter and mode in ['visible modality:', '']:
            res.append(T.RandomApply([T.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)], p=color_jitter_prob))
            logger.info("\tColorJitter: prob={}, brightness={}, contrast={}, saturation={}, hue={}".format(color_jitter_prob, brightness, contrast, saturation, hue))
            
        if do_gray and mode in ['visible modality:', '']:
            res.append(T.RandomGrayscale(p=gray_prob))
            logger.info("\tRandomGrayscale: prob={}".format(gray_prob))
            
        if do_vimc and mode in ['visible modality:', '']:
            res.append(T.RandomChoice([
                T.RandomApply([T.ColorJitter(hue=0.20)], p=0.5),
                ###### Visible-Infrared Modality Coordinator (VIMC) ######
                WeightedGrayscale(p=0.5),
                ChannelCutMix(p=0.5),
                SpectrumJitter(factor=(0.00, 1.00), p=0.5),
            ]))
            logger.info("\tVIMC: ColorJitter(hue=0.20), WeightedGrayscale(p=0.5), ChannelCutMix(p=0.5), SpectrumJitter(factor=(0.00, 1.00), p=0.5)")

        res.extend([T.ToTensor()])
        logger.info("\tToTensor")
        
        if do_rwa and mode in ['visible modality:', '']:
            res.append(RandomWeightedAverage(prob=rwa_prob))
            logger.info("\tRandomWeightedAverage: prob={}".format(rwa_prob))
            
        if do_mfb_rwa and mode in ['visible modality:', '']:
            res.append(MFBRandomWeightedAverage(img_size=size_train, N=mfb_rwa_N, prob=mfb_rwa_prob))
            logger.info("\tMFBRandomWeightedAverage: prob={}, N={}".format(mfb_rwa_prob, mfb_rwa_N))
            
        if do_cag and mode in ['visible modality:', '']:
            res.append(ChannelAdapGray(probability = cag_prob))
            logger.info("\tChannelAdapGray: prob={}".format(cag_prob))
            
        if do_norm:
            res.extend([T.Normalize(mean=mean_, std=std_)])
            logger.info("\tNormalize: mean={}, std={}".format(mean_, std_))

        if do_rea:
            res.append(T.RandomErasing(p=rea_prob, value=rea_value, scale=rea_scale, ratio=rea_ratio))
            logger.info("\tRandomErasing: prob={}, value={}, scale={}, ratio={}".format(rea_prob, rea_value, rea_scale, rea_ratio))

        if do_crea:
            res.append(ChannelRandomErasing(probability = crea_prob))
            logger.info("\tChannelRandomErasing: prob={}".format(crea_prob))
            
    else:
        logger.info("Test transform:")

        res.append(T.Resize(size=size_test, interpolation=InterpolationMode.BICUBIC))
        logger.info("\tResize: size={}".format(size_test))
        
        res.extend([T.ToTensor(), T.Normalize(mean=mean_, std=std_)])
        logger.info("\tToTensor, Normalize: mean={}, std={}".format(mean_, std_))
        
    return T.Compose(res)


def collate_fn(batch):  # img, label, cam_id, img_path, img_id
    samples = list(zip(*batch))

    data = [torch.stack(x, 0) for i, x in enumerate(samples) if i != 3]
    data.insert(3, samples[3])
    return data


def get_train_loader(args):
    dataset = args.dataset
    root = os.environ.get('DATA_ROOT', '/root/data/DataSets')
    p_size, k_size = args.trn_bs
    img_size = args.img_size
    num_workers = args.num_workers
    split_num = args.split_num
    
    if dataset == 'sysu':
        root = os.path.join(root, 'SYSU-MM01')
    elif dataset =='regdb':
        root = os.path.join(root, 'RegDB')
    elif dataset == 'llcm':
        root = os.path.join(root, 'LLCM')
    else:
        raise ValueError('Invalid dataset name: {}'.format(dataset))
    
    # data pre-processing
    transform = load_transforms(True, mode='',
                                mean_=args.pixel_mean, std_=args.pixel_std, size_train=img_size,
                                do_pad=args.pad_tf['do_pad'], padding_size=args.pad_tf['padding_size'], padding_mode=args.pad_tf['padding_mode'], padding_fill=args.pad_tf['padding_fill'],
                                do_flip=args.flip_tf['do_flip'], flip_prob=args.flip_tf['flip_prob'],
                                do_color_jitter=args.color_tf['do_color_jitter'], color_jitter_prob=args.color_tf['color_jitter_prob'], brightness=args.color_tf['brightness'], contrast=args.color_tf['contrast'], saturation=args.color_tf['saturation'], hue=args.color_tf['hue'],
                                do_gray=args.gray_tf['do_gray'], gray_prob=args.gray_tf['gray_prob'],
                                do_vimc=args.vimc_tf['do_vimc'],
                                do_rwa=args.rwa_tf['do_rwa'], rwa_prob=args.rwa_tf['rwa_prob'],
                                do_norm=args.norm_tf,
                                do_rea=args.rea_tf['do_rea'], rea_prob=args.rea_tf['rea_prob'], rea_value=args.rea_tf['rea_value'], rea_scale=args.rea_tf['rea_scale'], rea_ratio=args.rea_tf['rea_ratio'],
                                do_crea=args.crea_tf['do_crea'], crea_prob=args.crea_tf['crea_prob'],
                                do_cag=args.cag_tf['do_cag'], cag_prob=args.cag_tf['cag_prob'],
                                do_mfb_rwa=args.mfb_rwa_tf['do_mfb_rwa'], mfb_rwa_prob=args.mfb_rwa_tf['mfb_rwa_prob'], mfb_rwa_N=args.mfb_rwa_tf['mfb_rwa_N']
                                )
    # vis_transform = load_transforms(True, mode='visible modality:',
    #                             mean_=args.pixel_mean, std_=args.pixel_std, size_train=img_size,
    #                             do_pad=args.pad_tf['do_pad'], padding_size=args.pad_tf['padding_size'], padding_mode=args.pad_tf['padding_mode'], padding_fill=args.pad_tf['padding_fill'],
    #                             do_flip=args.flip_tf['do_flip'], flip_prob=args.flip_tf['flip_prob'],
    #                             do_color_jitter=args.color_tf['do_color_jitter'], color_jitter_prob=args.color_tf['color_jitter_prob'], brightness=args.color_tf['brightness'], contrast=args.color_tf['contrast'], saturation=args.color_tf['saturation'], hue=args.color_tf['hue'],
    #                             do_gray=args.gray_tf['do_gray'], gray_prob=args.gray_tf['gray_prob'],
    #                             do_vimc=args.vimc_tf['do_vimc'],
    #                             do_rwa=args.rwa_tf['do_rwa'], rwa_prob=args.rwa_tf['rwa_prob'],
    #                             do_norm=args.norm_tf,
    #                             do_rea=args.rea_tf['do_rea'], rea_prob=args.rea_tf['rea_prob'], rea_value=args.rea_tf['rea_value'], rea_scale=args.rea_tf['rea_scale'], rea_ratio=args.rea_tf['rea_ratio'],
    #                             do_crea=args.crea_tf['do_crea'], crea_prob=args.crea_tf['crea_prob'],
    #                             do_cag=args.cag_tf['do_cag'], cag_prob=args.cag_tf['cag_prob'],
    #                             do_mfb_rwa=args.mfb_rwa_tf['do_mfb_rwa'], mfb_rwa_prob=args.mfb_rwa_tf['mfb_rwa_prob'], mfb_rwa_N=args.mfb_rwa_tf['mfb_rwa_N']
    #                             )
    # inf_transform = load_transforms(True, mode='infrared modality:',
    #                             mean_=args.pixel_mean, std_=args.pixel_std, size_train=img_size,
    #                             do_pad=args.pad_tf['do_pad'], padding_size=args.pad_tf['padding_size'], padding_mode=args.pad_tf['padding_mode'], padding_fill=args.pad_tf['padding_fill'],
    #                             do_flip=args.flip_tf['do_flip'], flip_prob=args.flip_tf['flip_prob'],
    #                             do_color_jitter=args.color_tf['do_color_jitter'], color_jitter_prob=args.color_tf['color_jitter_prob'], brightness=args.color_tf['brightness'], contrast=args.color_tf['contrast'], saturation=args.color_tf['saturation'], hue=args.color_tf['hue'],
    #                             do_gray=args.gray_tf['do_gray'], gray_prob=args.gray_tf['gray_prob'],
    #                             do_vimc=args.vimc_tf['do_vimc'],
    #                             do_rwa=args.rwa_tf['do_rwa'], rwa_prob=args.rwa_tf['rwa_prob'],
    #                             do_norm=args.norm_tf,
    #                             do_rea=args.rea_tf['do_rea'], rea_prob=args.rea_tf['rea_prob'], rea_value=args.rea_tf['rea_value'], rea_scale=args.rea_tf['rea_scale'], rea_ratio=args.rea_tf['rea_ratio'],
    #                             do_crea=args.crea_tf['do_crea'], crea_prob=args.crea_tf['crea_prob'],
    #                             do_cag=args.cag_tf['do_cag'], cag_prob=args.cag_tf['cag_prob'],
    #                             do_mfb_rwa=args.mfb_rwa_tf['do_mfb_rwa'], mfb_rwa_prob=args.mfb_rwa_tf['mfb_rwa_prob'], mfb_rwa_N=args.mfb_rwa_tf['mfb_rwa_N']
    #                             )

    # dataset
    if dataset == 'sysu':
        train_dataset = SYSUDataset(root, mode='train', transform=transform)
    elif dataset == 'regdb':
        train_dataset = RegDBDataset(root, mode='train', transform=transform, split_num=split_num)
    elif dataset == 'llcm':
        train_dataset = LLCMDataset(root, mode='train', transform=transform)
        
    # sampler
    batch_size = p_size * k_size
    sampler = EvenlyRandomIdentitySampler(train_dataset, p_size * k_size, k_size)

    # loader
    train_loader = DataLoader(train_dataset, batch_size, sampler=sampler, drop_last=False, pin_memory=True,
                              collate_fn=collate_fn, num_workers=num_workers)
   
    return train_loader


def get_test_loader(args):
    dataset = args.dataset
    root = os.environ.get('DATA_ROOT', '/root/data/DataSets')
    batch_size = args.tst_bs
    img_size = args.img_size
    num_workers = args.num_workers
    split_num = args.split_num
    
    if dataset == 'sysu':
        root = os.path.join(root, 'SYSU-MM01')
    elif dataset =='regdb':
        root = os.path.join(root, 'RegDB')
    elif dataset == 'llcm':
        root = os.path.join(root, 'LLCM')
    else:
        raise ValueError('Invalid dataset name: {}'.format(dataset))
    
    # transform
    transform = load_transforms(False, 
                                mean_=args.pixel_mean, std_=args.pixel_std, size_test=img_size
                                )

    # dataset
    if dataset == 'sysu':
        gallery_dataset = SYSUDataset(root, mode='gallery', transform=transform)
        query_dataset = SYSUDataset(root, mode='query', transform=transform)
    elif dataset == 'regdb':
        gallery_dataset = RegDBDataset(root, mode='gallery', transform=transform, split_num=split_num)
        query_dataset = RegDBDataset(root, mode='query', transform=transform, split_num=split_num)
    elif dataset == 'llcm':
        gallery_dataset = LLCMDataset(root, mode='gallery', transform=transform)
        query_dataset = LLCMDataset(root, mode='query', transform=transform)
    
    
    # dataloader
    query_loader = DataLoader(dataset=query_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              pin_memory=True,
                              drop_last=False,
                              collate_fn=collate_fn,
                              num_workers=num_workers)

    gallery_loader = DataLoader(dataset=gallery_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                pin_memory=True,
                                drop_last=False,
                                collate_fn=collate_fn,
                                num_workers=num_workers)

    return gallery_loader, query_loader


def build_dataloaders(args):
    trn_dl = get_train_loader(args)
    gal_dl, que_dl = get_test_loader(args)
    
    data = [
        ['Train',   len(trn_dl.dataset.img_paths), trn_dl.dataset.num_ids, len(set(trn_dl.dataset.cam_ids))],
        ['Query',   len(que_dl.dataset.img_paths), que_dl.dataset.num_ids, len(set(que_dl.dataset.cam_ids))],
        ['Gallery', len(gal_dl.dataset.img_paths), gal_dl.dataset.num_ids, len(set(gal_dl.dataset.cam_ids))],
    ]
    table = tabulate(data, headers=['Split', '#Images', '#PIDs', '#CIDs'], tablefmt='grid')
    logger.info(f'# --- {args.dataset} Dataset --- #\n{table}')
    
    return trn_dl, gal_dl, que_dl