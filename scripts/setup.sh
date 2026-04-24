# med asr setup 
# conda create -n medasr python=3.10 -y
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate medasr
# pip install pandas
# pip install uv
# # install dependencies
# uv pip install git+https://github.com/huggingface/transformers.git@65dc261512cbdb1ee72b88ae5b222f2605aad8e5
# pip install -r requirements/requirements_medasr.txt

# # omilinugal asr setup
# conda create -n omnilingual python=3.10 -y
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate omnilingual
# pip install omnilingual-asr
# conda install -c conda-forge libsndfile==1.0.31 -y
# pip install torchaudio==2.8.0

# gemma model
# conda create -n gemma python=3.10 -y
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate gemma
# pip install -r requirements/requirements_gemma4.txt

# #  hugging face models
# conda create -n hf python=3.10 -y
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate hf
# pip install -U transformers torch accelerate

# for nvidia nemo models
# conda create -n nemo python=3.10 -y
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate nemo
# pip install -U nemo_toolkit['asr']
# pip install torchaudio


# qwen 3 Omni model 
# conda create -n qwen3 python=3.10 -y
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate qwen3
# pip install git+https://github.com/huggingface/transformers
# pip install accelerate
# pip install qwen-omni-utils -U
# pip install flash-attn --no-build-isolation
# conda install -c conda-forge libsndfile==1.0.31 -y

# conda create -n evaluation python=3.10 -y
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate evaluation
# pip install unbabel-comet
# pip install jiwer
# pip install numpy
# pip install pandas
# pip install openai
# pip install whisper_normalizer



