from itertools import product
import torch
import argparse
import numpy as np
import sys
import os
import math
import torchvision.datasets as tds
import torchvision.transforms as trans

#-------------------------------------------------------------------------------

ap = argparse.ArgumentParser(description='Convert dataset to LEO Mannix text source',formatter_class=argparse.RawTextHelpFormatter)
ap.add_argument('-ddn'  , metavar='<downloaded_dataset_name>' , type=str, default=None,  help='downloaded dataset name (e.g. FashionMNIST)')
ap.add_argument('-gdn'  , metavar='<generated_dataset_name>'  , type=str, default=None,  help='downloaded dataset name (e.g. mnist)')
ap.add_argument('-ni'   , metavar='<num_imgs>'                , type=str, default='all', help='Number of images convert (default all)')   
  
args = ap.parse_args()

#----------------------------------------------------------------------------------------------------------------

image_transform = trans.Compose([
  trans.Lambda(lambda image: torch.from_numpy(np.array(image).astype(np.float32)).unsqueeze(0))
])

if args.ddn : # Downloaded
   ds_access = getattr(tds,args.ddn)
   test_set = ds_access(root='./dataset/',train=False,download=True, transform=image_transform)
elif args.gdn : # Downloaded
# Loading custom Generated dataset
   workspace_path = './workspace/' 
   test_set_file_name =  workspace_path + args.gdn + '_ds_test.pt' 
   print("Loading Test dataset from: %s" % test_set_file_name)
   test_set= torch.load(test_set_file_name)['pt_ds'] 

else: 
   print('ERROR: Must provide attribute -ddn or -gdn')
   exit()
   
#----------------------------------------------------------------------------------------------------------------

if args.ddn  : # Downloaded
  np_dataset = test_set.data.numpy().astype(int)
  np_targets = test_set.targets.numpy().astype(int)

elif args.gdn : # Generated
  np_dataset = (test_set.tensors[0]).numpy().astype(int)
  np_targets = (test_set.tensors[1]).numpy().astype(int)
  
else: 
   print('ERROR: Must provide attribute -ddn or -gdn')
   exit()

ds_size = np_dataset.shape[0]

print('Size of Test data set:  %d' % ds_size)

#-----------------------------------------------------------------------------------------------------------------

img_num_row = np_dataset.shape[1] 
img_num_col  = np_dataset.shape[2] 

print('Detected image dimensions: %dX%d' % (img_num_row,img_num_col))

dsn = args.ddn if args.ddn else args.gdn
mnx_ds_txt_file_name =  'workspace/' + dsn + '_ds_mnx.txt'

mnx_ds_txt_file = open(mnx_ds_txt_file_name,'w')

#-----------------------------------------------------------------------------------------------------------------

test_idx = 0
for test_idx in range(ds_size) :

    img = np_dataset[test_idx] 
    label = np_targets[test_idx] 

    mnx_ds_txt_file.write('%02x %02x\n' % (test_idx,label))  

    img = np.maximum(0,np.minimum(255,img))  # Assumeand all values are in range of 0 to 255 (trunateif not)    
    for r in range(img.shape[0]) : 
      for c in range(img.shape[1]) :        
          mnx_ds_txt_file.write(' %02x' % int(img[r][c]))                        
      mnx_ds_txt_file.write('\n')
      
    mnx_ds_txt_file.write('\n')
    test_idx += 1
    if args.ni!='all' :
       if test_idx==int(args.ni) :
         break
      
mnx_ds_txt_file.close()

#----------------------------------------------------------------------------------------------------------------


