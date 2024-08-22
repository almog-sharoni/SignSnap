import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import QuantStub, DeQuantStub
import numpy as np

#-------------------------------------------------------------------------------

class mnx_descale(nn.Module): 

   def __init__(self,layer_name,dbg):
       super().__init__()
       self.dbg = dbg
       self.layer_name = layer_name

   def forward(self, input):  
       ret_vals = np.minimum(255.0,((input/256).type(torch.int64)).type(torch.float)) 
             
       if self.dbg :
          print("\nmnx_descale ret_vals for layer %s:\n:%s\n" % (self.layer_name,str(ret_vals))) # uncomment for debug
       return ret_vals
       
#-------------------------------------------------------------------------------

class rnn_in_reshape(nn.Module): 

   def __init__(self,tcpt):
       super().__init__()
       self.tcpt = tcpt # if True Assume RNN input tensor is with columns per time , otherwise it is with row per time.

   def forward(self, input): 
       if self.tcpt  :     
         ret_tensor = input.reshape(input.shape[0],input.shape[1]*input.shape[2],input.shape[3])
         ret_tensor = ret_tensor.transpose(2,1)
       else :
        # This is equivalent to horizontally stacking the sub 2d arrays of the 3d array (within a 4d tensor)
        ret_tensor = torch.permute(input,(0,2,1,3)).reshape(input.shape[0],input.shape[2],input.shape[1]*input.shape[3])    
       return ret_tensor

#-------------------------------------------------------------------------------

class extract_rnn_out_tensor(nn.Module):
    def __init__(self):
       super().__init__()
    def forward(self,input):
        # Output shape (batch, features, hidden)
        tensor, _ = input
        # Reshape shape (batch, hidden)
        # print('DBG %s\n' % str(tensor[:, -1, :]))
        return tensor[:, -1, :]

#-------------------------------------------------------------------------------

def add_seq_cmn_layers(seq_layers=None, layer_idx=None, is_conv=False, 
                       out_channels=False, add_bn=False, add_relu=False, 
                       add_dropouts=False, add_mnx=False, dbg=False) :

    prfx_str = 'conv2d' if is_conv else 'fc'
    
    if add_bn :
      if is_conv : 
        seq_layers.add_module('%s_bn_%d'%(prfx_str,layer_idx) ,nn.BatchNorm2d(out_channels)) 
      else: # Default is FC
        seq_layers.add_module('%s_bn_%d'%(prfx_str,layer_idx) ,nn.BatchNorm1d(out_channels))       
    if add_relu :
      seq_layers.add_module('%s_relu_%d'%(prfx_str,layer_idx) ,nn.ReLU())                      
    if  (add_dropouts!=0) : # dropouts layer
      seq_layers.add_module('%s_dop_%d'%(prfx_str,layer_idx) , nn.Dropout(add_dropouts))                    
    if add_mnx :
        layer_name = '%s_mnx_descale_%d'%(prfx_str,layer_idx)
        seq_layers.add_module(layer_name, mnx_descale(layer_name=layer_name,dbg=dbg))

#-------------------------------------------------------------------------------

class res(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return self.module(inputs) + inputs

#-----------------------------------------------------------------------------------

class custom_rnn(torch.nn.Module):

   def __init__(self,num_layers,input_size,hidden_size,mnx_mode) :
        super().__init__()
        self.num_layers = num_layers
        self.input_size = input_size        
        self.hidden_size = hidden_size 
        self.mnx_mode = mnx_mode

        self.weight_ih = []        
        self.weight_hh = []
        
        def set_param(layer) :    
            layer_input_size = self.input_size if layer==0 else self.hidden_size         
            ih = nn.Parameter(torch.Tensor(self.hidden_size,layer_input_size))
            hh = nn.Parameter(torch.Tensor(self.hidden_size,self.hidden_size))
            self.weight_ih.append(ih)
            self.weight_hh.append(hh)
            return ih,hh
        
        for layer in range(num_layers) :
          # TODO: find a more elegant and generic way for below (without undesired use of 'exec')
          if    layer==0 : self.weight_ih_l0,self.weight_hh_l0 = set_param(layer)
          elif  layer==1 : self.weight_ih_l1,self.weight_hh_l1 = set_param(layer)
          elif  layer==2 : self.weight_ih_l2,self.weight_hh_l2 = set_param(layer)
          elif  layer==3 : self.weight_ih_l3,self.weight_hh_l3 = set_param(layer)
          elif  layer==4 : self.weight_ih_l4,self.weight_hh_l4 = set_param(layer)
          elif  layer==5 : self.weight_ih_l5,self.weight_hh_l5 = set_param(layer)
          elif  layer==6 : self.weight_ih_l6,self.weight_hh_l6 = set_param(layer)
          elif  layer==7 : self.weight_ih_l7,self.weight_hh_l7 = set_param(layer)
          elif  layer==8 : self.weight_ih_l8,self.weight_hh_l8 = set_param(layer)
          elif  layer==9 : self.weight_ih_l9,self.weight_hh_l9 = set_param(layer)
          else : 
            print('ERROR: Currently supporting only up to 9 RNN layers, quitting')
            exit()
   #---------------------------------------------------------------------      
     
   # Forward Initially Based on https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
   # Notice Bias handling excluded not supported in mnx mode
   
   def forward(self, input, h_0=None): 
 
     layer_in = input[0] # remove batch dimension
     seq_len,_ = layer_in.size()
     
     for layer in range(self.num_layers):  
        
        if (layer>0) :
           layer_in = layer_out
           
        h_0 = torch.zeros(self.hidden_size)
        h_t_prev = h_0
        h_t = h_0
        output = []        
        
        weight_ih_hh_cnct = torch.concatenate((self.weight_ih[layer].T, self.weight_hh[layer].T)) 
        
        for t in range(seq_len):
        
             in_prev_cnctnt = torch.concatenate((layer_in[t],h_t_prev))             
             h_t = torch.relu(in_prev_cnctnt @ weight_ih_hh_cnct) # vector,matrix mul            
              
             if (self.mnx_mode): # divide by 256, same as mannix HW
               h_t = torch.clamp((h_t/256).type(torch.int64),max=255).type(torch.float) 
                
             output.append(h_t)
             h_t_prev = h_t
             
        layer_out = torch.stack(output)
     
     # Wrap with a batch dimension of 1 (applicable only for inference for single image per iterating invocation)
     output = torch.Tensor(1,layer_out.size()[0],layer_out.size()[1])
     output[0] = layer_out
              
     return output, h_t

#-----------------------------------------------------------------------------------

class nn_model(nn.Module):
    def __init__(self, args, num_conv_ch, fc_dim, device, apply_bn=True, ncl=2, pool=True, nfl=2, pad=None, cdop=0, ldop=0, ris=0, mnx=False,cusr=False, dbg=False):
        
        # NN architecture, param output_dim: dimension on the categories

        super().__init__()
        
        # TODO : move all args arguments to transferred args object.
        
        self.apply_bn = apply_bn
        self.mnx = mnx
        self.custom_rnn = cusr        
        self.device = device
        self.ncl = ncl
        self.pool = pool 

        # print('DBG ncl = %d , pool = %s' % (ncl,str(pool)))
        # exit()
        
        self.nfl = nfl
        self.pad = pad
        self.cdop = cdop 
        self.ldop = ldop      
        self.num_conv_ch = num_conv_ch
        self.nds = args.nds
        self.nhcc = int(args.nhcc)
        self.grp = args.grp
        self.ris = ris # rnn input size
        self.fc_dim = fc_dim
      
        self.layers = nn.Sequential()
        
        do_lin_bias  = not args.nlb
        do_conv_bias = not args.ncb        
        do_rnn_bias  = not args.nrb
                
        if (self.nds==0) : # CONV Layers
             for i in range(self.ncl) :      
                in_channels = self.num_conv_ch[i] 
                out_channels = self.num_conv_ch[i+1]
                if (args.cgr!='NA') :
                  groups = args.cgr[i]
                else :         
                  groups = 1 if (i==0) else  self.grp             
                kernel_size = 5
                if self.pool :
                   padding = 0 
                else :
                   padding=self.pad[i] 
                                                    
                self.layers.add_module('conv2d_%d'%i ,nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,padding=padding,groups=groups,bias=do_conv_bias))
       
                add_seq_cmn_layers(seq_layers=self.layers, layer_idx=i, is_conv=True, out_channels=out_channels, 
                                   add_bn=self.apply_bn, add_relu=True, add_dropouts=self.cdop, add_mnx=self.mnx, dbg=dbg)
                                   
                if self.pool :
                   self.layers.add_module('pool_%d'%i ,nn.MaxPool2d(kernel_size=2, stride=2))
            
        else  : # DS Layers
 
                ds_layers = nn.Sequential()
 
                for s in range(self.nds) :

                    # DW Layer
                    i = 2*s     
                    in_channels = 1 if i==0 else self.nhcc
                    out_channels = self.nhcc
                    kernel_size = 5             
                    padding = 0 if i==(ncl-2) else 2  # Don't pad on last DW,PW pair
                    groups = 1 if i==0 else self.nhcc
                    padding=self.pad[i]  
                   
                    ds_layers.add_module('conv2d_%d'%i ,nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,padding=padding,groups=groups,bias=do_conv_bias))                    

                    add_seq_cmn_layers(seq_layers=ds_layers, layer_idx=i, is_conv=True, out_channels=out_channels, 
                                       add_bn=self.apply_bn, add_relu=True, add_dropouts=self.cdop, add_mnx=self.mnx, dbg=dbg)

                    # PW Layer
                    i=2*s+1
                    in_channels =  self.nhcc
                    out_channels = 1 if i==(ncl-1) else self.nhcc  
                    kernel_size = 1
                    groups = 1           
                    padding=self.pad[i]  
                  
                    ds_layers.add_module('conv2d_%d'%i ,nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,padding=padding,groups=groups,bias=do_conv_bias))                    

                    add_seq_cmn_layers(seq_layers=ds_layers, layer_idx=i, is_conv=True, out_channels=out_channels, 
                                       add_bn=self.apply_bn, add_relu=True, add_dropouts=self.cdop, add_mnx=self.mnx, dbg=dbg)                
                                
                if args.rds :
                  self.layers.append(res(ds_layers))
                else :
                  self.layers.append(ds_layers)                  

        if args.nrl!=0 : # Optionally add RNN conv output before FC layers
               
           self.layers.add_module('rnn_in_reshape' , rnn_in_reshape(tcpt=args.tcpt))

           if not (self.mnx or self.custom_rnn) :  
              self.layers.add_module('rnn', nn.RNN(
                input_size    = self.ris    , # Number of expected features in the input x
                hidden_size   = args.rhs    , # Number of features in the hidden state h
                num_layers    = args.nrl    , # Number of recurrent layers 
                nonlinearity  = 'relu'      , # Can be either 'tanh' or 'relu'
                bias          = do_rnn_bias , # If False, then the layer does not use bias weights b_ih and b_hh.
                batch_first   = True        , # ??? If True, then the input,output tensors provided as (batch,seq,feature) instead of (seq,batch,feature). 
                dropout       = self.ldop   , # If non-zero, introduces a Dropout layer on the outputs of each RNN
                bidirectional = args.rbid   , # If True, becomes a bidirectional RNN.            
              )) 
           else : 
              self.layers.add_module('rnn',custom_rnn(num_layers=args.nrl,input_size=self.ris,hidden_size=args.rhs,mnx_mode=self.mnx))
           
           self.layers.add_module('extract_rnn_out_tensor',extract_rnn_out_tensor())
 
        # Flat before FC Layer       
        self.layers.add_module('flat_conv_out' , nn.Flatten()) 

        # FC Layers        
        for i in range(self.nfl) :   
        
           self.layers.add_module('fc%d'%i, nn.Linear(in_features=self.fc_dim[i], out_features=self.fc_dim[i+1],bias=do_lin_bias))

           add_relu = (self.ldop!=0) and (i!=self.nfl)
           add_seq_cmn_layers(seq_layers=self.layers, layer_idx=i, is_conv=False, out_channels=self.fc_dim[i+1], 
                               add_bn=self.apply_bn, add_relu=add_relu, add_dropouts=self.ldop, add_mnx=self.mnx, dbg=dbg)
          
    def forward(self, input_img):
              
        # return self.layers(input_img)
        return nn.Sequential(*self.layers)(input_img)

 