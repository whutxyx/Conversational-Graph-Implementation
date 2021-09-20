
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

PYPATH=/home/yuanxin/WorkSpace/projects/Dialog/miniconda3/envs/paddle_py37/bin

CUDA_VISIBLE_DEVICES=3 $PYPATH/python run.py --infer \
--model_dir dongming_test_output/20210917/MMPMS-041609/model_epoch_18 \
--num_mappings 5 \
--result_file dongming_test_output/20210917/result_18

#CUDA_VISIBLE_DEVICES=3 $PYPATH/python eval.py dongming_test_output/20210917/result

