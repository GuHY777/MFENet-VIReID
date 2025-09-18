from utils.registry import Registry
import logging

logger = logging.getLogger(__name__)

MODEL_REGISTRY = Registry("MODEL")

from .vmamba_vi import VMambaVI
from .agw_vi import AGW_VI
from .mpanet import MPANet
from .fdnm import FDNM
from .fff import FFF
from .mpanet2 import MPANet2
from .caj import CAJ
from .baseline import BaseLine
from .deen import DEEN
from .baseline2 import BaseLine2
from .baseline3 import BaseLine3
from .baseline4 import BaseLine4
from .mfenet import MFENet
from .mfenet_no2 import MFENet_no2


def build_model(args, trn_dl):
    logger.info("\n# --- Model --- #")
    logger.info(f'\tKwargs={args.model_kwargs}')
    
    if 'use_sie' in args.model_kwargs:
        use_sie = args.model_kwargs['use_sie']
    else:
        use_sie = 0
    if use_sie == 1:
        num_sies = len(set(trn_dl.dataset.cam_ids))
    elif use_sie == 2:
        num_sies = 2
    else:
        num_sies = 0
    
    return MODEL_REGISTRY[args.model](num_classes=trn_dl.dataset.num_ids, dataset=args.dataset, img_size=args.img_size, num_sies=num_sies, **args.model_kwargs)