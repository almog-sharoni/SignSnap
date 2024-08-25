cd signSnap/SignSnap/leomnx_user_space/apps/SignSnap/pt

python3 gen_SignSnap_ptds.py 

python3 signSnap_train.py 

python3 sign_lang_inference.py -wfn workspace/sign_lang_tmw.pt -nt 5000 

python3 egn_mnx_model_params_file.py -wfn workspace/sign_lang_tmw.pt -sapl 0.8 -qpt 

python3 sign_lang_inference.py -wfn workspace/sign_lang_tmw_mnx_params.pt -nt 5000 -mnx