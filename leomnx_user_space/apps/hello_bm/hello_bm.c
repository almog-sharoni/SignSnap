#include <leo2_libs.h>

int main() {

  leo_2_init() ;

  bm_printf("\nHELLO! This application demonstrate some useful baremetal library services\n");   

  unsigned int ex_inF         = bm_fopen_r("app_src_dir/bm_example_in.txt") ;
  unsigned int ex_hex_inF     = bm_fopen_r("app_src_dir/bm_example_hex_in.txt") ;  
  unsigned int ex_outF        = bm_fopen_w("app_src_dir/bm_example_out.txt") ;
  unsigned int ascii_art_inF  = bm_fopen_r("app_src_dir/helllo_bm_ascii_art.txt") ;
  
  char str_buf[80] ;
  char IS_EOF = 0 ;
  
  IS_EOF = !bm_fgets(ex_inF,str_buf);
  bm_printf("\nGot hex value string %s from input file\n",str_buf) ;
  
  unsigned int val = hex_str_to_uint(str_buf) ;
  
  bm_printf("Decimal value of string %s is %d\n",str_buf,val);  
  
  bm_fprintf(ex_outF,"DEC(%s)=%d\n",str_buf,val); 
  
  IS_EOF = !bm_fgets(ex_inF,str_buf);
  bm_printf("\nGot decimal value string %s from input file\n",str_buf) ;
  
  signed int num = dec_str_to_int(str_buf) ;
  bm_printf("num=%d\n",num);
  
  bm_load_ascii_file(ex_inF,str_buf,num);  
  
  bm_printf("Loaded following ASCII characters from input file\n");
  for (int i=0;i<num;i++) bm_printf("%c",str_buf[i]);
  bm_printf("\n");
  
  bm_printf("Dumping %d characters to output file\n",num);
  bm_dump_ascii_file(ex_outF,str_buf,num);

  #define VEC_SIZE 5
  unsigned char bytes_vec1[VEC_SIZE] ;
  unsigned char bytes_vec2[VEC_SIZE] ; 
  
  bm_load_hex_file (ex_hex_inF,VEC_SIZE,bytes_vec1); // load vec1 byte from hex file
  bm_load_hex_file (ex_hex_inF,VEC_SIZE,bytes_vec2); // load vec2 byte from hex file 
  
  bm_printf("vec1 =    ["); for (int i=0;i<VEC_SIZE;i++) bm_printf("%2x ",bytes_vec1[i]); bm_printf("]\n");
  bm_printf("vec2 =    ["); for (int i=0;i<VEC_SIZE;i++) bm_printf("%2x ",bytes_vec2[i]); bm_printf("]\n");

  unsigned char max_vec[VEC_SIZE] ; 
  for (int i=0;i<VEC_SIZE;i++) max_vec[i] = bytes_vec1[i] > bytes_vec2[i]  ? bytes_vec1[i] : bytes_vec2[i] ;
  bm_printf("max_vec = ["); for (int i=0;i<VEC_SIZE;i++) bm_printf("%2x ",max_vec[i]); bm_printf("]\n");    
  
  bm_printf("Also dumping max_vec to output file\n");
  bm_dump_hex_file (ex_outF, (char *)max_vec, VEC_SIZE);
  
  bm_printf("Measuring cycle count to calculate dot-product of vec1 and vec2\n");
  
  
  int start_cycle,end_cycle ; 
  unsigned int dot_prod = 0 ;  
  
  ENABLE_CYCLE_COUNT ; // Enable the cycle counter
  RESET_CYCLE_COUNT  ; // Reset counter to ensure 32 bit counter does not wrap in-between start and end. 
  
  GET_CYCLE_COUNT_START(start_cycle) ;   
  for (int i=0;i<VEC_SIZE;i++) dot_prod += (unsigned int)bytes_vec1[i] * (unsigned int)bytes_vec2[i]  ;
  GET_CYCLE_COUNT_END(end_cycle) ;
  int cycle_cnt = end_cycle-start_cycle ;
  
  bm_printf("dot_prod = %d (decimal)\n",dot_prod);
  bm_printf("Took %d cycles to calculate dot-prod of %d byte elements size\n\n",cycle_cnt,VEC_SIZE);
    
  while (!IS_EOF) {
   IS_EOF = !bm_fgetln (ascii_art_inF,str_buf) ;
   bm_printf("%s",str_buf) ;
  }
 
  bm_fclose(ex_inF);
  bm_fclose(ex_hex_inF);  
  bm_fclose(ex_outF);   
  bm_fclose(ascii_art_inF);  
  
  bm_quit_app();  
  return 0;
}
