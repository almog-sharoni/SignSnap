import sys
import argparse
import torch
import numpy as np
import torchvision.datasets as tds
import torchvision.transforms as trans
sys.path.append('../../../../leo_mannix/pymnx/')
import pt_train as ptt

# Dataset Download, Will download if does not exist
# see info here: https://pytorch.org/vision/0.18/generated/torchvision.datasets.FashionMNIST.html

# This dataset transformation keeps the image as grey scal 0 to 255 value (oppose to trans.toTensor which normlize them to 0 to 1 float.)
image_transform = trans.Compose([
  trans.Lambda(lambda image: torch.from_numpy(np.array(image).astype(np.float32)).unsqueeze(0))
])

train_set = tds.FashionMNIST(root='./dataset/',train=True, download=True, transform = image_transform) 
val_set   = tds.FashionMNIST(root='./dataset/',train=False,download=True, transform = image_transform) 

workspace_path = './workspace/'

ap = argparse.ArgumentParser(description='pt train',formatter_class=argparse.RawTextHelpFormatter)
ap.add_argument('-scf'   , metavar='<scf_conf_name>' , type=str, help='Test source file name')
args = ap.parse_args()

ptt.invoke(workspace_path = workspace_path, scf=args.scf , train_set=train_set, val_set=val_set)
