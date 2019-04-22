# ./dist_train.sh <job> <index>

CUDA_VISIBLE_DEVICES="" \
python sync_train.py \
     --ps_hosts=localhost:2222 \
     --worker_hosts=localhost:2223,localhost:2224 \
     --job_name=$1 --task_index=$2
