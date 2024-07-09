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
I=1
JOB_NAME=eval_gpt2_vanilla_295B_2M_$I
PYTHONPATH=${PROJECT_PATH}:${PYTHONPATH} torchrun --nproc_per_node 8 --master_addr $MASTER_ADDR --master_port $MASTER_PORT --nnodes 1 --node_rank 0  train.py \
	config/eval_gpt2_2M_big.py

# interactive base gpt peerification
PYTHONPATH=${PROJECT_PATH}:${PYTHONPATH} torchrun --nproc_per_node 8 --master_addr $MASTER_ADDR --master_port $MASTER_PORT --nnodes 1 --node_rank 0  train.py \
	config/peerify_base_11_4.py

# interactive gpt peerification tabulation step
PYTHONPATH=${PROJECT_PATH}:${PYTHONPATH} torchrun --nproc_per_node 8 --master_addr $MASTER_ADDR --master_port $MASTER_PORT --nnodes 1 --node_rank 0  train.py \
	config/peerify_base_11_3_tabulate.py

# interactive gpt peerification fullvqization step 4
PYTHONPATH=${PROJECT_PATH}:${PYTHONPATH} torchrun --nproc_per_node 8 --master_addr $MASTER_ADDR --master_port $MASTER_PORT --nnodes 1 --node_rank 0  train.py \
	config/peerify_base_11_4_full.py

# interactive gpt peerification fullvqization step 1
PYTHONPATH=${PROJECT_PATH}:${PYTHONPATH} torchrun --nproc_per_node 8 --master_addr $MASTER_ADDR --master_port $MASTER_PORT --nnodes 1 --node_rank 0  train.py \
	config/peerify_base_11_1_full.py

# interactive gpt peerification fullvqization 16 options step 1
PYTHONPATH=${PROJECT_PATH}:${PYTHONPATH} torchrun --nproc_per_node 8 --master_addr $MASTER_ADDR --master_port $MASTER_PORT --nnodes 1 --node_rank 0  train.py \
	config/peerify_base_11_1_16_full.py

# interactive gpt peerification fullvqization 16 options 8 heads step 1
PYTHONPATH=${PROJECT_PATH}:${PYTHONPATH} torchrun --nproc_per_node 8 --master_addr $MASTER_ADDR --master_port $MASTER_PORT --nnodes 1 --node_rank 0  train.py \
	config/peerify_base_11_1_16_full_8.py

# interactive gpt peerification fullvqization 8 options step 1
PYTHONPATH=${PROJECT_PATH}:${PYTHONPATH} torchrun --nproc_per_node 8 --master_addr $MASTER_ADDR --master_port $MASTER_PORT --nnodes 1 --node_rank 0  train.py \
	config/peerify_base_11_1_8_full.py

# interactive big model hf test run
I=0
JOB_NAME=train_gpt2_vanilla_295B_2M_hf_$I
PYTHONPATH=${PROJECT_PATH}:${PYTHONPATH} torchrun --nproc_per_node 8 --master_addr $MASTER_ADDR --master_port $MASTER_PORT --nnodes 1 --node_rank 0  train.py \
	config/train_gpt2_2M_big_hf.py

# interactive vanilla hf push to hub
python push_to_hub.py out/gpt2-vanilla-295B-2M-hf/ckpt_150000.pt gpt2-owt-295B

# interactive lm eval invocation
lm_eval --model hf \
    --model_args pretrained=pbelcak/gpt2-owt-295B \
    --tasks hellaswag,openbookqa,commonsense_qa,piqa,social_iqa,winogrande,arc_easy,mmlu \
    --device cuda:0 \
    --batch_size 8 \
    --limit 1000 \
    --output_path $NANO/out/gpt2-vanilla-295B-2M-hf/results.json

# a keepalive test
python $PB/keepalive/keepalive.py add --job=train_gpt2_2M_big_hf_owt_sbb --startswith --indicator=$NANO/out/train_gpt2_2M_big_hf_owt_sbb/.DONE --command="$NANO/cluster/run_job_big.sh config/train_gpt2_2M_big_hf_owt_sbb.py"