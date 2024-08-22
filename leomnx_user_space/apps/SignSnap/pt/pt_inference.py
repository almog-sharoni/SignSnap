import argparse
import torch
import numpy as np
import sys
import os
import pt_nn_model as nnm

#------------------------------------------------------------------------------------------------------------
# @torch.no_grad()
#--------------------------------------------------------------------------------

class DictObj:
  def __init__(self, in_dict:dict):
        assert isinstance(in_dict, dict)
        for key, val in in_dict.items():
            if isinstance(val, (list, tuple)):
              setattr(self, key, [DictObj(x) if isinstance(x, dict) else x for x in val])
            else:
              setattr(self, key, DictObj(val) if isinstance(val, dict) else val)
               
#-------------------------------------------------------------------------------

class infer():
 
    def __init__(self,args):
        
        self.device = torch.device("cpu")  
                
        print("Loading model params from: %s" % args.wfn)
        self.loaded_dict = torch.load(args.wfn, map_location=self.device)      
              
        if args.mnx:
          train_wfn = args.wfn[:args.wfn.find('_mnx')] + '.pt'
          print("Loading train reference dataset just for configuration info from: %s" % args.wfn)
          self.ref_loaded_dict = torch.load(train_wfn, map_location=self.device)
        else :
           self.ref_loaded_dict = self.loaded_dict

        self.train_args         = DictObj(self.ref_loaded_dict['train_args'])        
        self.num_conv_ch        = self.ref_loaded_dict['num_conv_ch']
        self.fc_dim             = self.ref_loaded_dict['fc_dim']   
        self.apply_bn           = self.ref_loaded_dict['apply_bn'] 
        self.ris                = self.ref_loaded_dict['ris']          
        self.pad                = self.ref_loaded_dict['pad']
        self.cdop               = self.ref_loaded_dict['cdop']         
        self.ldop               = self.ref_loaded_dict['ldop']         
        self.ncl                = self.ref_loaded_dict['ncl']  
        self.pool               = self.ref_loaded_dict['pool']         
        self.nfl                = self.ref_loaded_dict['nfl']         
        self.label_names_by_idx = self.ref_loaded_dict['lnbi']     
        
        if args.sbn :
          self.apply_bn = False
           
        self.model = nnm.nn_model(args=self.train_args,num_conv_ch=self.num_conv_ch,fc_dim=self.fc_dim,device=self.device,apply_bn=self.apply_bn, 
                            ncl=self.ncl, pool=self.pool, nfl=self.nfl, pad=self.pad, cdop=self.cdop, ldop=self.ldop, ris=self.ris, mnx=args.mnx,cusr=args.cusr,dbg=args.dbg).to(self.device)        

        ignore_missing_keys = args.sbn and self.train_args.bn
        strict_load = not ignore_missing_keys
        
        if args.mnx:
          self.model.load_state_dict(self.loaded_dict,strict=strict_load)        
        else:
          self.model.load_state_dict(self.loaded_dict['model_state'],strict=strict_load)
                             
    #--------------------------------------------------------------------------------
    
    def check_inference_on_test_dataset(self,args,test_set) : 
          
        print("Loading test dataset")
  
        num_tests =  int(args.nt)
        num_correct = 0
        num_error = 0
                                         
        # if self.apply_bn :
        #    self.model.eval()
        
        self.model.eval() # For now always apply model.eval() in inference
        
        print('Testing %d cases' % num_tests)    
        for i in range(num_tests) :

            test_idx = i  
                
            if self.train_args.tdl : # Tourchvision Downloaded dataset
                                                      
                test_img = test_set[test_idx][0][0]
                target   = test_set[test_idx][1]
            
            else:
            
                test_img = test_set['pt_ds'].tensors[0][test_idx] 
                target   = test_set['pt_ds'].tensors[1][test_idx]
                               
            test_img_tensor = torch.FloatTensor(1,(test_img.size())[0],(test_img.size())[1])            
            test_img_tensor[0] = test_img
                                                                              
            sis = test_img_tensor.size()
            
            # print('DBG %s' % str(test_img))
            # exit() # DBG        
                        
            test_img_tensor = test_img_tensor.reshape(sis[0],1,sis[1],sis[2])
            pred = self.model(test_img_tensor)
            
            pred_np = pred.detach().numpy()       
            pred_idx = np.argmax(pred_np)
              
            pred_ok = (pred_idx==target)  
            if pred_ok :            
              num_correct+=1           
            else :               
              num_error+=1                                          
            
            if np.all(pred_np.astype(int)==int(0)) :  # Indicate all 0
              detect_str = 'Undetected: All output vector items are zeros'
            elif np.all(pred_np.astype(int)==int(255)) : # Indicate all 0xf (255)
              detect_str = 'Undetected: All output vector items are 0xff'
            elif np.all(pred_np.astype(int) == pred_np.astype(int)[0][0]) : # Indicate all same (but not 0 or 255)  
              detect_str = 'Undetected: All same (but not 0 or 0xff)'              
            else :
              if pred_ok :
                detect_str = 'OK idx %d (\'%s\')'%(pred_idx,self.label_names_by_idx[pred_idx]) 
              else : 
                detect_str = 'Miss idx %d (\'%s\') instead of %d (\'%s\')' %(pred_idx,self.label_names_by_idx[pred_idx],target,self.label_names_by_idx[target])                 

            # print ('%s' % detect_str)
                                           
            num_total = num_correct + num_error   
            print('%-50s  ; num_correct %d (%2.1f%%) , num_error %d (%2.1f%%) , total %d' %  
            (detect_str, num_correct, 100*num_correct/num_total, num_error, 100*num_error/num_total, num_total))
                    
    #--------------------------------------------------------------------------------------

if False : # Currently mot supporting direct invocation

    if __name__ == '__main__':
    
        ap = argparse.ArgumentParser(description='Inference',formatter_class=argparse.RawTextHelpFormatter)  
        
        ap.add_argument('-wfn'   , metavar='<weights_src_name>' , type=str, help='Model weights source file name') 
        ap.add_argument('-tfn'   , metavar='<test_src_fn>'      , type=str, help='Test source file name')
        ap.add_argument('-nt'    , metavar='<num_tests>'        , type=str, help='Number of Tests')
        ap.add_argument('-sbn'   , action='store_true'                    , help='Skip Batch-Normalization (override model driven')  
        ap.add_argument('-dbg'   , action='store_true'                    , help='Enable some debug prints')        
        ap.add_argument('-mnx'   , action='store_true'                    , help='Apply Mannix Descale') 
        
           
        args = ap.parse_args()
    
        infer = infer(args)
       
        infer.check_inference_on_test_dataset(args)  
