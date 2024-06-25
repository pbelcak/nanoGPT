# interactive dbg sesh
srun --account=nvr_lpr_llm --partition=interactive,grizzly,polar,polar2,polar3,polar4  --time=04:00:00 --container-image nvcr.io/nvidia/pytorch:23.11-py3 -n 1 --gpus 8 --cpus-per-gpu 16 --container-mounts=$HOME:/home,/lustre:/lustre --pty /bin/bash
# THEN
cd $NANO
source cluster/prepare_job.sh
source cluster/secrets.sh

PROJECT_PATH=$NANO

# interactive train lut run
I=0
JOB_NAME=train_lut_6A_$I
PYTHONPATH=${PROJECT_PATH}:${PYTHONPATH} torchrun --nproc_per_node 8 --master_addr $MASTER_ADDR --master_port $MASTER_PORT --nnodes 1 --node_rank 0  train.py \
	config/train_gpt2_lut_6A.py

# interactive train fs run
I=0
JOB_NAME=train_fs_6A_$I
PYTHONPATH=${PROJECT_PATH}:${PYTHONPATH} torchrun --nproc_per_node 8 --master_addr $MASTER_ADDR --master_port $MASTER_PORT --nnodes 1 --node_rank 0  train.py \
	config/train_gpt2_fs_6C.py

# interactive vanilla gpt eval
I=0
JOB_NAME=eval_gpt2_vanilla_295B_2M_$I
PYTHONPATH=${PROJECT_PATH}:${PYTHONPATH} torchrun --nproc_per_node 8 --master_addr $MASTER_ADDR --master_port $MASTER_PORT --nnodes 1 --node_rank 0  train.py \
	config/eval_gpt2_2M_big.py

# interactive base gpt peerification
I=0
JOB_NAME=peerify1_base_$I
PYTHONPATH=${PROJECT_PATH}:${PYTHONPATH} torchrun --nproc_per_node 8 --master_addr $MASTER_ADDR --master_port $MASTER_PORT --nnodes 1 --node_rank 0  train.py \
	config/peerify1_base.py