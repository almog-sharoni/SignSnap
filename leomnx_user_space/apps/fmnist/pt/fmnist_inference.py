import sys
import argparse
import torchvision.datasets as tds
import torchvision.transforms as trans
import torch
import numpy as np
import random
sys.path.append('../../../../leo_mannix/pymnx/')
import pt_inference as pti


# For reproducibility - Does not seem to do the job, TODO Investigate 
# TODO: consider controlling from config file parameter
# torch.manual_seed(0) 
# np.random.seed(0)
# random.seed(10)

# Dataset Test Download, Will download if does not exist
# see info here: https://pytorch.org/vision/0.18/generated/torchvision.datasets.FashionMNIST.html


# This dataset transformation keeps the image as grey scal 0 to 255 value (oppose to trans.toTensor which normlize them to 0 to 1 float.)
image_transform = trans.Compose([
  trans.Lambda(lambda image: torch.from_numpy(np.array(image).astype(np.float32)).unsqueeze(0))
])

test_set   = tds.FashionMNIST(root='./dataset/',train=False,download=True, transform=image_transform)

workspace_path = './workspace/'

ap = argparse.ArgumentParser(description='Squiggle Inference',formatter_class=argparse.RawTextHelpFormatter)  

ap.add_argument('-wfn'   , metavar='<weights_src_name>' , type=str, help='Model weights source file name') 
ap.add_argument('-nt'    , metavar='<num_tests>'        , type=str, help='Number of Tests')
ap.add_argument('-sbn'   , action='store_true'                    , help='Skip Batch-Normalization (override model driven')  
ap.add_argument('-dbg'   , action='store_true'                    , help='Enable some debug prints') 
ap.add_argument('-mnx'   , action='store_true'                    , help='Apply Mannix Descale') 
ap.add_argument('-cusr'  , action='store_true'                    , help='Custom rnn mode')  
   
args = ap.parse_args()

infer = pti.infer(args)

infer.check_inference_on_test_dataset(args,test_set)  


