import torch
import logging
from collections import defaultdict
import os.path as osp
from utils import AverageMeter
import numpy as np
from torch.nn.utils import clip_grad_norm_
from data.freqaugs import FreqAug



logger = logging.getLogger(__name__)


class Engine:
    def __init__(self, args, model, model_ema, optim, lrs, loss, evaluator, dl_trn, tb_writer, savedir) -> None:
        self.args = args
        self.model = model
        self.model_ema = model_ema
        self.optim = optim
        self.lrs = lrs
        self.loss = loss
        self.evaluator = evaluator
        self.dl_trn = dl_trn
        self.tb_writer = tb_writer
        self.savedir = savedir
        self.best_mAP = 0.0
        
        if self.args.amp:
            logger.info("\nUsing Automatic Mixed Precision (AMP)")
            self.scaler = torch.cuda.amp.GradScaler(2**(10.0)) # 2**(10.0)
            
        if len(self.args.eval_freq) == 1:
            self.eval_epochs = [self.args.eval_freq[0] for _ in range(self.args.epochs // self.args.eval_freq[0])]
            self.eval_epochs =  np.cumsum(self.eval_epochs).tolist()
            if self.args.epochs % self.args.eval_freq[0]:
                self.eval_epochs.append(self.args.epochs)
        elif len(self.args.eval_freq) == 3: # [20, 100, 1] --> [20,40,60,80,100,101,102,...]
            self.eval_epochs = [self.args.eval_freq[0] for _ in range(self.args.eval_freq[1] // self.args.eval_freq[0])]
            self.eval_epochs =  np.cumsum(self.eval_epochs).tolist()
            if self.args.eval_freq[1] % self.args.eval_freq[0]:
                self.eval_epochs.append(self.args.eval_freq[1])
            eval_epochs = np.arange(self.args.eval_freq[1], self.args.epochs, self.args.eval_freq[2]).astype(int).tolist()
            self.eval_epochs.extend(eval_epochs)
            if self.eval_epochs[-1] != self.args.epochs:
                self.eval_epochs.append(self.args.epochs)
        else:
            assert self.args.eval_freq[-1] == self.args.epochs
            self.eval_epochs = self.args.eval_freq
            
        self.epoch = 0
        
        # TODO Fixed images for visualization
        if hasattr(self.model, 'eval_hooks'):
            fixed_imgs = []
            for imgs, _, _, _, _ in self.evaluator.dls[0]:
                fixed_imgs.append(imgs[::32])
                break
            for imgs, _, _, _, _ in self.evaluator.dls[1]:
                fixed_imgs.append(imgs[::32])
                break
            fixed_imgs = torch.cat(fixed_imgs, dim=0)
            fixed_imgs = fixed_imgs.to('cuda')
            self.fixed_imgs = fixed_imgs
        # TODO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
        # self.aug = FreqAug(prob=0.5, ratio=1.0, alpha=0.5, rwa_prob=1.0, img_size=(288, 144), only_rgb_aug=False)
        
         
    def train_one_epoch(self, show_nums=50):
        # TODO Fixed images for visualization
        if hasattr(self.model, 'eval_hooks'):
            self.model.eval_hooks(self.fixed_imgs, self.tb_writer, self.epoch)
        # TODO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
        # * freeze or unfreeze the backbone
        if self.args.freeze_bb and self.epoch == 0 and hasattr(self.model, 'freeze_backbone'):
            self.model.freeze_backbone()
            logger.info("Backbone is frozen")
        
        if self.args.freeze_bb and self.epoch == self.args.freeze_bb and hasattr(self.model, 'unfreeze_backbone'):
            self.model.unfreeze_backbone()
            logger.info("Backbone is unfrozen")
            
        if self.args.eval_bb and self.epoch == 0 and hasattr(self.model, 'eval_backbone'):
            # assert self.args.eval_bb <= self.eval_epochs[0], "eval_bb should be less than or equal to the first eval_freq!"
            self.model.eval_backbone()
            logger.info("Backbone is in eval mode")
        
        if self.args.eval_bb and self.epoch == self.args.eval_bb and hasattr(self.model, 'train_backbone'):
            self.model.train_backbone()
            logger.info("Backbone is in train mode")
            
        losses_avg = AverageMeter()
        gradnorm_avg = AverageMeter()
        
        for i, (imgs, pids, cids, _, _) in enumerate(self.dl_trn):
            
            imgs = imgs.to('cuda')
            pids = pids.to('cuda')
            cids = cids.to('cuda')
            
            # imgs = self.aug(imgs)
            
            with torch.autocast(device_type='cuda', enabled=self.args.amp):
                outputs = self.model(imgs, cids)                             
                loss_val, losses = self.loss(outputs, pids)  
            losses_avg(losses)
            
            if torch.isnan(loss_val):
                # the loss is NaN, which means the model has produced an invalid output
                return False
            
            if self.args.amp:
                self.scaler.scale(loss_val).backward()
                self.scaler.unscale_(self.optim)
                if self.args.grad_clip > 0:
                    norm = clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                    gradnorm_avg({'gradnorm': norm.item()})
                else:
                    norm = 0.0

                self.scaler.step(self.optim)
                self.scaler.update()
            else:
                loss_val.backward()
                if self.args.grad_clip > 0:
                    norm = clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                    gradnorm_avg({'gradnorm': norm.item()})
                else:
                    norm = 0.0

                self.optim.step()
            self.optim.zero_grad()
            
            if self.model_ema is not None:
                self.model_ema.update(self.model)
            
            if (i + 1) % show_nums == 0:
                logger.info(f"\tIter [{i+1}] Losses: {losses_avg} Gradnorm: {gradnorm_avg}")
                # raise ValueError("Stopped by me")
                
        if self.args.lr_scheduler == 'TimmScheduler':
            lrs = self.lrs._get_lr(self.epoch)[:2]
        else:
            lrs = self.lrs.get_last_lr()[:2]
        if self.tb_writer is not None:
            self.tb_writer.add_scalars('lrs', {'lr-'+str(i):lr for i,lr in enumerate(lrs)}, self.epoch+1)

        logger.info(f"\tIter [{i+1}] Losses: {losses_avg} Gradnorm: {gradnorm_avg}")
        logger.info(f"Epoch [{self.epoch+1}/{self.args.epochs}] Lrs {[f'{lrs[i]:.4e}' for i in range(len(lrs))]} Losses: {losses_avg}")
        if self.tb_writer is not None:
            self.tb_writer.add_scalars('losses', losses_avg.avgs, self.epoch+1)

        return True
    
    def train_test(self):
        logger.info("Start training...")
        
        start_epoch = self.epoch
        self.model.train()
        # ! get the initial uncertainty factor
        # uncertainty_factor0 = -1.0
        # for m in self.model.modules():
        #     if isinstance(m, LowPerturb3D):
        #         uncertainty_factor0 = m.uncertainty_factor
        
        # if hasattr(self.model, 'get_tau'):
        #     tau = self.model.get_tau()
        
        for self.epoch in range(start_epoch, self.args.epochs):
            # ! set the factor in lowperturb factor
            # if self.epoch < 20:
            #     if hasattr(self.model, 'set_lowperturb_factor'):
            #         self.model.set_lowperturb_factor(0.0)
                
            #     if hasattr(self.model, 'set_tau'):
            #         self.model.set_tau(tau)
            # elif self.epoch < 120:
            #     if hasattr(self.model, 'set_lowperturb_factor'):
            #         self.model.set_lowperturb_factor(uncertainty_factor0 / 100.0 * (self.epoch - 19))
                    
            #     if hasattr(self.model, 'set_tau'):
            #         self.model.set_tau(tau - (self.epoch - 9) * (tau - 1.0) / 100.0)
            
            # if hasattr(self.model, 'hooks_enabled') and self.model.hooks_enabled and self.epoch % 1 == 0 and hasattr(self.model, 'hooks_vis'):
            #     self.model.hooks_vis(self.epoch, self.tb_writer)
            
            # with torch.autograd.detect_anomaly():
            if not self.train_one_epoch(self.args.show_nums):
                raise ValueError("NaN loss encountered")
            
            if self.args.lr_scheduler == 'TimmScheduler':
                self.lrs.step(self.epoch)
            else:
                self.lrs.step()

            if (self.epoch+1) in self.eval_epochs:
                _mAP = self.test(self.epoch)[1]
                if _mAP > self.best_mAP:
                    self.best_mAP = _mAP
                    logger.info(f"Best mAP: {_mAP:.4f}")
                    self.save_checkpoint(osp.join(self.savedir, f"best_ckpt.pth"))
                
            # if self.epoch+1 == self.eval_epochs[-1]:
            #     self.save_checkpoint(osp.join(self.savedir, f"ckpt_{self.epoch+1}.pth"))
        if self.model_ema is not None:
            torch.save(self.model_ema.ema.state_dict(), osp.join(self.savedir, f"ema_model.pth"))
    
    def test(self, epoch=-1, ckpt=None):
        if ckpt is not None:
            if 'model' in ckpt:
                self.model.load_state_dict(ckpt['model'])
            else:
                self.model.load_state_dict(torch.load(ckpt))
        metrics = self.evaluator(epoch+1)
        self.model.train()
        
        if self.args.eval_bb and self.epoch < self.args.eval_bb and hasattr(self.model, 'eval_backbone'):
            # assert self.args.eval_bb <= self.eval_epochs[0], "eval_bb should be less than or equal to the first eval_freq!"
            self.model.eval_backbone()
            logger.info("Backbone is in eval mode")
        
        return metrics
    
    def save_checkpoint(self, checkpoint_dir: str):
        ckpt_path = osp.join(checkpoint_dir, "ckpt.pth") if '.pth' not in checkpoint_dir else checkpoint_dir
        ckpt = {
            'epoch': self.epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optim.state_dict(),
            'lr_scheduler': self.lrs.state_dict(),
            'scaler': self.scaler.state_dict() if self.args.amp else None
        }
        torch.save(ckpt, ckpt_path)
        
    def load_checkpoint(self, checkpoint_dir: str):
        ckpt_path = osp.join(checkpoint_dir, "ckpt.pth") if '.pth' not in checkpoint_dir else checkpoint_dir
        ckpt = torch.load(ckpt_path)
        self.epoch = ckpt['epoch']
        self.model.load_state_dict(ckpt['model'])
        self.optim.load_state_dict(ckpt['optimizer'])
        self.lrs.load_state_dict(ckpt['lr_scheduler'])
        if self.args.args.amp and ckpt['scaler'] is not None:
            self.scaler.load_state_dict(ckpt['scaler'])
