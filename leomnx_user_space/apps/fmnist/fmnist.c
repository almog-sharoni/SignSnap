#include <leo2_libs.h>
#include "mannix_main.h" 

//-------------------------------------------------------------------------------------------------------


int main() {
          
 leo_2_init() ;    

 bm_printf("HELLO FASHION MNIST\n"); 
 
 // Specifying the desired model parameters
 #if GP_DEF==CRNN // This is effected by "-gpd CRNN" in invocation
   char* model_config_file = "app_src_dir/pt/workspace/fmnist_crnn_tmw_mannix_model_config.txt";
   char* model_params_file = "app_src_dir/pt/workspace/fmnist_crnn_tmw_model_params_mfdb.txt"; 
 #else
   char* model_config_file = "app_src_dir/pt/workspace/fmnist_conv_tmw_mannix_model_config.txt";
   char* model_params_file = "app_src_dir/pt/workspace/fmnist_conv_tmw_model_params_mfdb.txt";
 #endif
 
 // Specifying the desired dataset
 char* dataset_file = "app_src_dir/pt/workspace/FashionMNIST_ds_mnx.txt";
   
 mannix_main(model_config_file, model_params_file, dataset_file) ;
 
 bm_quit_app();  
 return 0;

}

//-------------------------------------------------------------------------------------------------------

// report_detection will be called by mannix_main after each detection iteration  
// You may either call back the generic report or provide your custom reporting code.
// To apply the generic mechanism invoke with -ccd CUSTOM_REPORT , or permanently set to defined.

void report_detection(int real_label, int detected_label, char aborted) {
    
   #ifndef CUSTOM_REPORT 

    generic_report_detection(real_label, detected_label, aborted) ;

   #else 
    
   static int ok_cnt = 0 ;
   static int total_cnt = 0 ;
   const char* class_str[] = {"Tshirt","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Boot"};   

   total_cnt++ ;
   if (detected_label==real_label) ok_cnt++;   
   bm_printf("%3.1f%% ", ((float)(100*ok_cnt))/(float)total_cnt) ; 

   if (detected_label==real_label) bm_printf("%11s OK\n",class_str[detected_label]) ;
   else bm_printf("%11s ERR (%s)\n",class_str[detected_label],class_str[real_label]) ;

     
   #endif
       
    
}