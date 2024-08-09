PROJECT_PATH=$PB/nanoGPT

# VANILLA GPT2
I=0
JOB_NAME=gpt2_$I
./cluster/run_job.sh \
	config/train_gpt2.py

# GPT2 with LUT at 11th layah
I=0
JOB_NAME=gpt2_lut_$I
./cluster/run_job.sh \
	config/train_gpt2_lut.py

# GPT2 with A-LUT at 6-11th layah
I=0
JOB_NAME=gpt2_lut_6A_${I}
./cluster/run_job.sh \
	config/train_gpt2_lut_6A.py

# GPT2 with B-LUT at 6-11th layah
I=0
JOB_NAME=gpt2_lut_6B_${I}
./cluster/run_job.sh \
	config/train_gpt2_lut_6B.py

# insert and tune-in tabular moe (pte)
./cluster/run_job.sh \
	config/peerify_base_11_tabularmoe.py

# insert and tune-in small mlp
./cluster/run_job.sh \
	config/peerify_base_11_smallmlp512.py

# absorb any loss in predictive performance by other neural components
./cluster/run_job.sh \
	config/peerify_base_11_absorb.py