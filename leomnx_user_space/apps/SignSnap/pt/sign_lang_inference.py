import sys
import argparse
import torchvision.datasets as tds
import torchvision.transforms as trans
import torch
import numpy as np
sys.path.append('../../../../leo_mannix/pymnx/')
import pt_inference as pti

# Loading custom Generated dataset
workspace_path = './workspace/' 
test_set_file_name =  workspace_path + 'sign_lang_test.pt' 

print("Loading Test dataset from: %s" % test_set_file_name)
test_set= torch.load(test_set_file_name) 

workspace_path = './workspace/'

ap = argparse.ArgumentParser(description='sign_lang Inference',formatter_class=argparse.RawTextHelpFormatter)  

ap.add_argument('-wfn'   , metavar='<weights_src_name>' , type=str, help='Model weights source file name') 
ap.add_argument('-nt'    , metavar='<num_tests>'        , type=str, help='Number of Tests')
ap.add_argument('-sbn'   , action='store_true'                    , help='Skip Batch-Normalization (override model driven')  
ap.add_argument('-dbg'   , action='store_true'                    , help='Enable some debug prints') 
ap.add_argument('-mnx'   , action='store_true'                    , help='Apply Mannix Descale') 
ap.add_argument('-cusr'  , action='store_true'                    , help='Custom rnn mode')
  
args = ap.parse_args()

infer = pti.infer(args)

infer.check_inference_on_test_dataset(args,test_set)  


