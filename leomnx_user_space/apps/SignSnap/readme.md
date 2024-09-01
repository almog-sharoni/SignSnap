cd signSnap/SignSnap/leomnx_user_space/apps/SignSnap/pt

python3 gen_SignSnap_ptds_qscale_augs.py  --augment_factor 10

python3 signSnap_train.py 

python3 sign_lang_inference.py -wfn workspace/sign_lang_tmw.pt -nt 5000 

python3 gen_mnx_model_params_file.py -wfn workspace/sign_lang_tmw.pt -sapl 1.0 -qpt 

python3 sign_lang_inference.py -wfn workspace/sign_lang_tmw_mnx_params.pt -nt 5000 -mnx

python3 ds_pt_to_mnx.py -gdn sign_lang