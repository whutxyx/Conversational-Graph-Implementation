
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

PYPATH=/home/yuanxin/WorkSpace/projects/Dialog/miniconda3/envs/paddle_py37/bin

#$PYPATH/python preprocess.py

CUDA_VISIBLE_DEVICES=2,3,4 $PYPATH/python run.py --data_dir data --batch_size 2 --infer_batch_size 2 --save_dir test_output



