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
