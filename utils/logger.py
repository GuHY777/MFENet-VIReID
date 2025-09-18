import logging
import os
import sys
import shutil
from datetime import datetime


def create_file(exp, basedir='./logs'):
    if not os.path.exists(basedir):
        os.mkdir(basedir)
        
    # Get the current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    folder_name = f"{(timestamp)}-------{exp}"  # Create a folder name with the experiment name and timestamp    
    os.mkdir(os.path.join(basedir, folder_name))
    return os.path.join(basedir, folder_name)


def setup_logger(exp, load_cfg='', basedir='./runs', fileout=True, savecodes=True):
    savedir =create_file(exp, basedir)
    
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # stdout logging: master only
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s", datefmt="%m/%d %H:%M:%S")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # file logging: all workers
    if fileout:
        plain_formatter = logging.Formatter(
                "[%(asctime)s] %(name)s <<%(levelname)s>>: %(message)s", datefmt="%m/%d %H:%M:%S"
                )
        filename = os.path.join(savedir, "log.txt")
        fh = logging.FileHandler(filename, 'w')
        fh.setLevel(logging.INFO)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)
        
    if savecodes:
        shutil.copytree(os.getcwd(), os.path.join(savedir, 'code'), ignore=shutil.ignore_patterns('runs', '__pycache__', '*.pth','.history','.vscode','.model_cache'))
    
    logger.info(f'Experiment \'{exp}\' started, saving to \'{savedir}\'.') 
    if load_cfg:    
        logger.info(f'Loading config from \'{load_cfg}\'.')
    return logger, savedir



