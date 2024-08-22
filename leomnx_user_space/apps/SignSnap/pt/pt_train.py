from scipy import signal
from itertools import product
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import torch.nn.functional as F
import numpy as np
import argparse
import copy
import sys
from sys import exit
import os
import shutil
import gzip
import math
from importlib import import_module

import pt_nn_model as nnm

#-------------------------------------------------------------------------------

def count_label_classes(dataset=None,args=None,quite=True):

    MAX_CLASS_INT_LABEL = 40
    cl_vec_max = np.zeros((MAX_CLASS_INT_LABEL,), dtype=int) 
    
    # count 
    label_classes = set()

    if args.tdl  : 
      for _, label in dataset:
          label_item = label       
          cl_vec_max[label_item] += 1
          label_classes.add(label_item)
    else :
      for label in dataset.tensors[1]:        
          cl_vec_max[label.item()] += 1
          label_classes.add(label.item())
    
    num_classes = len(label_classes)
    cl_vec = cl_vec_max[:num_classes]
    if not quite: 
      print('items per class : %s' % str(cl_vec))
    
    # get weights
    total_samples = len(dataset)
    class_weights = total_samples / (cl_vec)
    if not quite: 
      print('total_samples = %d' % total_samples)      
      print('class_weights = %s' % str(class_weights))        
    
    weights = []
    for _, label in dataset:                  
        label_item = label if args.tdl else label.item()   
        weights.append(class_weights[label_item])    
    
    return num_classes, weights 

#-------------------------------------------------------------------------------

def get_num_correct(preds, sample_labels):
    
    return preds.argmax(dim=1).eq(sample_labels).sum().item()

#--------------------------------------------------------------------------------

def pred_sample(sample,model,device,args,is_eval):

       sample_images,sample_labels = sample
       sis = sample_images.size()
                     
       if not args.tdl : 
          sample_images = sample_images.reshape(sis[0],1,sis[1],sis[2])
                          
       sample_labels = sample_labels.to(device)
       
       if not is_eval:
         model.train()       
         preds = model(sample_images.to(device)).to(device)
       else :
         model.eval()
         preds = model(sample_images.to(device)).to(device)

       return preds,sample_labels

#---------------------------------------------------------------------------------

def train(model, train_set, val_set, learn_rate, num_of_epoch, batch_size, device,args, num_train, num_val, weights):
    model = model.to(device)
    # optimizer = optim.Adam(model.parameters(), lr=learn_rate)
    
    # see https://towardsdatascience.com/weight-decay-and-its-peculiar-effects-66e0aee3e7b8 for weight_decay recommendation
    # weight_decay = 0.01
    weight_decay = 0.05 * math.sqrt(batch_size/(num_of_epoch*num_train))
    print('Applying L2 Regularization with weight_decay = %f' % weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=learn_rate, weight_decay=weight_decay) # Checking effect of L2 Regularization
     
    shuffle_mode = False # Experimenting True/False , notice sampler option is mutually exclusive with shuffle
    
    for epoch in range(num_of_epoch):
        total_loss = 0.
        total_correct = 0.
        all = 0.
 
        # Notice dataset is not label balanced, this should compensate.
        weighted_sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
                
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle_mode, sampler=weighted_sampler)
        
        for sample in train_loader:
                               
            optimizer.zero_grad()
            
            preds,sample_labels = pred_sample(sample=sample,model=model,device=device,args=args,is_eval=False)
                                               
            loss = F.cross_entropy(preds,sample_labels).to(device)

            loss.backward()
                       
            optimizer.step()
            
            total_loss += loss.item()

            total_correct += get_num_correct(preds, sample_labels)
                        
            all += batch_size

        val_accuracy = accuracy(model, val_set ,device, args, num_val)

        print('epoch %d , train accuracy: %3.2f , total loss: %2.1f ; validate accuracy = %3.2f' % (epoch,total_correct/all,total_loss, val_accuracy))

#------------------------------------------------------------------------------------------------------

def accuracy(model, val_set, device, args, num_val):

    # the network accuracy calculated by comparing the predictions and the true sample_labels

    total = 0 
    total_correct = 0
    shuffle_mode = False # Experimenting True/False , notice sampler option is mutually exclusive with shuffle
    
    batch_size = int(0.01*num_val) if ((args.bs=='calc') or (int(100*float(args.tp))==0))  else int(float(args.bs)*float(args.tp))

    # Notice dataset is not always label balanced, this should compensate.
    num_val_classes, val_weights = count_label_classes(dataset=val_set,args=args,quite=True)
    val_weighted_sampler = WeightedRandomSampler(val_weights, num_samples=len(val_weights), replacement=True)
    
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=shuffle_mode, sampler=val_weighted_sampler)
    
    for sample in val_loader:  
    
        preds,sample_labels = pred_sample(sample=sample,model=model,device=device,args=args,is_eval=True)
 
        total += batch_size
        total_correct += get_num_correct(preds, sample_labels) 
    
    accuracy_val = total_correct / total

    return accuracy_val

#---------------------------------------------------------------------------------------------------------

def parse_ccc(args) : # explicit configuration of conv layers num channel out if provided

    # legal argument format example : ',6,4,2,1' 

    if args.ccc == 'NA' :
      return None
    ccc_split_str = (args.ccc).split(',')
    ccc_len = len(ccc_split_str)
    if len(ccc_split_str) != int(args.ncl) :
       print('ERROR: ccc argument does not match number of conv layers (ncl)')
       exit()
    ccc = []
    for i in range(ccc_len) :
      try :
        ccc.append(int(ccc_split_str[i]))
      except : 
        print('ccc argument illegal format') 
        exit() 
      if (i<(ccc_len-1)) and (ccc[-1] > 6) and args.cgr=='NA':
        print('ERROR inner num channels max val is 6')
        exit()      
        
    return ccc     
#---------------------------------------------------------------------------------------------------------


def parse_cgr(args,ccc) : # explicit grouping for ccc

    # legal argument format example : ',6,4,2,1' 

    if args.cgr == 'NA' :
      cgr = [1]*len(ccc)
      return cgr
    cgr_split_str = (args.cgr).split(',')
    cgr_len = len(cgr_split_str)
    if len(cgr_split_str) != int(args.ncl) :
       print('ERROR: cgr argument does not match number of conv layers (ncl)')
       exit()
    cgr = []
    for i in range(cgr_len) :
      try :
        cgr.append(int(cgr_split_str[i]))      
      except : 
        print('cgr argument illegal format') 
        exit()   
 
      num_in_ch = 1 if i==0 else ccc[i-1]
      if  not (num_in_ch/cgr[i]).is_integer() :
        print('ERROR: Illegal custom ccc,cgr configuration, per layer number of channels of both input and output divided by its cgr should be an integer')
        exit()
        
    return cgr     

#---------------------------------------------------------------------------------------------------------

def run_train(args, train_set, val_set) : 

    # check if there are GPU available fot this run
    cuda = torch.cuda.is_available()
    if cuda :
       print('Device is GPU/CUDA')
    else: 
       print('Device is CPU')    
    
    device = torch.device("cuda" if cuda else "cpu")
    nhcc = int(args.nhcc)
    nlcc = int(args.nlcc)
    
    # Regarding groups constrains see for example the discussion here:
    # https://discuss.pytorch.org/t/understanding-goup-in-conv2d/34026/3
    
    if not (args.nds!='0') : # Currently not supporting groups for DS 
      for num_ch in [nhcc,nlcc] :
        grp_div = num_ch/args.grp   
        if grp_div.is_integer() :
          grp_div = int(grp_div)
        else:
         print('ERROR: args nhcc,nlcc divided by grp must be an integer')   
         exit()
         
    if (args.nds==0) and (nhcc/args.grp > 6) :
     print('ERROR: args nhcc,nlcc divided by grp must be an integer not bigger than 6')
     exit()
                    
    if args.rds :   
        print('Notice: in rds mode padding is forced to be 2')
        args.pad = '2'        

    if args.spl : # Split train,validate from master dataset
    
       # fetch the data for the network
       dataset_loaded_dict = torch.load(args.fsn)
       dataset = dataset_loaded_dict['pt_ds']
       
       lnbi = dataset_loaded_dict['lnbi']
       
       num_classes,weights = count_label_classes(dataset=dataset,args=args,quite=True)       
       print('Detected %d unique labeled classes in dataset' % num_classes)       
       print('Total number of tests  %d' % dataset.tensors[0].size()[0])
       
       ds_size = dataset.tensors[0].size()[0]
       
       if (int(100*float(args.tp))==0): # Mostly for debug purposes   
          num_train = ds_size
          num_val = num_train*0.1       
          train_set = dataset
          val_set = train_set     
       else :
          num_val = int(ds_size*float(args.tp))       
          num_train = ds_size - num_val  
          train_set, test_set = torch.utils.data.random_split(dataset, [num_train, num_val])
          val_set = test_set # TMP!!! Using same

       img_num_row = dataset.tensors[0].size()[1]
       img_num_col = dataset.tensors[0].size()[2] 
       
    elif args.tdl : # Tourchvision Downloaded dataset
           
       num_train = train_set.data.size()[0]
       num_val = val_set.data.size()[0]
              
       num_classes,train_weights = count_label_classes(dataset=train_set,args=args,quite=True)       
       print('Detected %d unique labeled classes in dataset' % num_classes)       
       print('Total number of train images %d' % num_train)
       
       img_num_row = train_set.data.size()[1] 
       img_num_col  = train_set.data.size()[2]  
       
       train_set.classes
       lnbi = [] # label names by idx
       for ds_class in train_set.classes :
         lnbi.append(ds_class)
         
    else: # load train,validate from pre-split custom made dataset

       num_train = train_set.tensors[0].size()[0]
       num_val = val_set.tensors[0].size()[0]
       
       num_classes,train_weights = count_label_classes(dataset=train_set,args=args,quite=True)       
       print('Detected %d unique labeled classes in dataset' % num_classes)       
       print('Total number of train images %d' % num_train)
       
       lnbi = args.lnbi                       
       img_num_row = train_set.tensors[0].size()[1]  
       img_num_col = train_set.tensors[0].size()[2] 
                     
    # Fixed parameters (per this model) 
    NUM_OF_EPOCH = int(args.nep)
    
    print('Input Image structure : img_num_row=%d , img_num_col=%d' % (img_num_row,img_num_col))
    
    cnn_k = 5
    ncl = None # Num conv layers 

    args.nds = int(args.nds)
    if (args.nds!=0) :
        args.ncl  = str(args.nds*2)   # pairs of depth wise amd Pointwise layers
   
    ncl = int(args.ncl)
    nfl = int(args.nfl)    

    cdop = float(args.cdop) # conv dropout
    ldop = float(args.ldop) # lin dropout
        
    pad = []
    num_conv_ch  = [] # number of conv layers interfacing input channels
    # indexing: [0] is input of first layer, [1] is num output ch of layer[0] and input of layer[1] etc.
    num_conv_ch.append(1)

    ccc = parse_ccc(args) # explicit configuration of conv layers num channel out if provided   
    if ccc!=None :
       args.cgr = parse_cgr(args,ccc) # explicit configuration of cc groups  
    
    if ccc==None : 
       for i in range(1,ncl) :
            num_conv_ch.append(nhcc) # default is 6 
       num_conv_ch.append(nlcc)  #  num ch out of last conv layer
    else :
       for i in range(1,ncl+1) :
         num_conv_ch.append(ccc[i-1])  # Notice num_conv_ch include the first layer input num channels which is always 1 
       
    if args.pad=='mrp' : # auto add minimal required padding for given kernel
      conv_out_x = img_num_col
      conv_out_y = img_num_row
      for i in range(ncl) :
        padh,padw = 0,0      
        if conv_out_y < 5 :
          padh = int(np.ceil((cnn_k-conv_out_y)/2))

        if conv_out_x < 5 :
          padw = int(np.ceil((cnn_k-conv_out_x)/2))

        pad.append((padh,padw)) # TODO works well only for odd input dimensions   
        conv_out_y = conv_out_y-(cnn_k-1)+(2*padh)          
        conv_out_x = conv_out_x-(cnn_k-1)+(2*padw) 
        print('conv2d[%d] : conv_out_y=%d , conv_out_x=%d' % (i,conv_out_y,conv_out_x))        

    else:  
      
      if  args.pool and (int(args.pad)==0) and (args.pad!='mrp') and (args.nds==0) :
          conv_out_x = img_num_col
          conv_out_y = img_num_row
          for i in range(ncl) :            
            conv_out_y = (conv_out_y-(cnn_k-1))//2       
            conv_out_x = (conv_out_x-(cnn_k-1))//2  

            if (conv_out_x==0) or (conv_out_y==0)  :
                print('ERROR: Num Convolution Layers (ncl) with pool option reduce intermediate feature-map dimensions below 5X5 kernel feasibility')
                exit()


            print('conv2d[%d] after pool_max_2x2 : conv_out_y=%d , conv_out_x=%d' % (i,conv_out_y,conv_out_x))        

      else :
          pad_val = int(args.pad)       
          num_padded_layers = ncl if (args.nds==0) else args.nds         
          conv_per_dim_reduce =  num_padded_layers*((cnn_k-1)-(pad_val*2)) 
          conv_out_x = img_num_col - conv_per_dim_reduce            
          conv_out_y = img_num_row - conv_per_dim_reduce
          if (conv_out_x==0) or (conv_out_y==0)  :
            print('ERROR: Num Convolution Layers (ncl) reduce intermediate feature-map dimensions below 5X5 kernel feasibility')
            exit()
          
          if (pad_val>0) :
            print('num padded layers=%d,conv_per_dim_reduce=%d, conv_out_x=%d , conv_out_y=%d' % (num_padded_layers,conv_per_dim_reduce,conv_out_x,conv_out_y))               
          for i in range(ncl) :
            if (args.nds>0) and (i%2)==0 :  # in DS mode pad only even layers (depthwise)
                 pad_val = 0
            pad.append((pad_val,pad_val))
    
    
    if (args.nds!=0) :
      fc0_dim = conv_out_x * conv_out_y 
      ris = conv_out_y  # rnn input_size
    elif (args.nrl!=0) : 
      if args.tcpt :
         ris = num_conv_ch[-1] * conv_out_y  # features in columns
      else :
         ris = num_conv_ch[-1] * conv_out_x  # features in rows

      if (args.rhs =='calc') :
          args.rhs = ris
          print('Calculated rhs (rnn hidden size) %d' % ris)
         
      fc0_dim = args.rhs * (2 if args.rbid else 1)
    else: 
      fc0_dim = num_conv_ch[-1] * conv_out_x * conv_out_y 
      ris = 0      
    

    OUTPUT_DIM = num_classes 
   
    fc_dim_auto = np.zeros((nfl+1,), dtype=int) 
    fc_dim_auto[0] = fc0_dim
    
    ratio_per_fc_layer = (fc0_dim/OUTPUT_DIM)**(1/nfl)
    print('ratio_per_fc_layer=%f' % ratio_per_fc_layer)   

    print('fc_dim[0]=%d' % fc0_dim)    
    for i in range(1,nfl) : 
       fc_dim_auto[i] = int((fc_dim_auto[i-1])/ratio_per_fc_layer)
       print('fc_dim[%d]=%d' % (i, fc_dim_auto[i])) 
    fc_dim_auto[nfl] = OUTPUT_DIM   
    print('OUTPUT_DIM=%d' % (OUTPUT_DIM))
   
    batch_size = int(0.01*num_train) if args.bs=='calc'  else int(args.bs)
    learn_rate = float(args.lr)        
    
    # train the model
    print("\nBATCH_SIZE=%d, fc_dim=%s, learn_rate=%f" % (batch_size, str(fc_dim_auto), learn_rate))
                  
    nn_model = nnm.nn_model(args=args,num_conv_ch=num_conv_ch,fc_dim=fc_dim_auto,device=device,apply_bn=args.bn, 
                            ncl=ncl, pool=args.pool, nfl=nfl, pad=pad, cdop=cdop, ldop=ldop, ris=ris, mnx=False,dbg=False).to(device)        

    # num_train_classes, train_weights = count_label_classes(dataset=train_set,args=args,quite=True)

    train(model=nn_model, train_set=train_set, val_set=val_set, learn_rate=learn_rate, 
         num_of_epoch=NUM_OF_EPOCH, batch_size=batch_size, device=device, args=args, num_train=num_train, num_val=num_val, weights=train_weights)

    current_accuracy = accuracy(nn_model, val_set ,device, args, num_val)
    print("validate accuracy: %3.2f" % current_accuracy)
   
    # save parameters of generated model to be loaded for inference
    model_params_file_name = args.workspace_path + args.dbn + '_tmw.pt'
    
    # to be included in dict for analysis etc.
    args.img_num_col  = img_num_col
    args.img_num_row  = img_num_row
    
    torch.save(dict(model_state=nn_model.state_dict(),
    train_args=args.__dict__, num_conv_ch=num_conv_ch, pool=args.pool, fc_dim=fc_dim_auto, apply_bn=args.bn, ncl=ncl, nfl=nfl, pad=pad, cdop=cdop, ldop=ldop, ris=ris, lnbi=lnbi),model_params_file_name)   
  
#----------------------------------------------------------------------------------------------------------------

def get_scf(args) :

   if args.scf != None :
       class args_obj(object) : pass
       scf_args = args_obj()
       sys.path.append('./scf')
       conf = import_module(args.scf)
       conf.set_args(scf_args)
       print('Reading configuration args from %s' % args.scf)
       try :
        sys_argv = sys.argv
       except :
        sys_argv = []       
       for attr,value in scf_args.__dict__.items() :
          if (str(attr) not in sys_argv) and ('-%s'%str(attr) not in sys.argv) :        
             print ("%s : %s" % (str(attr), str(value)))          
             args.__dict__[attr] = value
          else :
             print ("%s : %s ; Overwritten by command line" % (str(attr), str(args.__dict__[attr])))             
          
#----------------------------------------------------------------------------------------------------------------

def invoke(workspace_path=None, scf=None,train_set=None, val_set=None,ap=None) :

    argparse.ArgumentParser(description='pt train',formatter_class=argparse.RawTextHelpFormatter)
           
    class args_obj(object) : 
      pass
    args = args_obj()
    args.scf = scf
    args.workspace_path = workspace_path
    
    get_scf(args)
    
    if not args.tdl :
      args.lnbi = train_set['lnbi']    
      train_set = train_set['pt_ds']
      val_set = val_set['pt_ds'] 
    
    run_train(args,train_set,val_set)   
    
  