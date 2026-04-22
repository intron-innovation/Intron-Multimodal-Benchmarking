# med asr setup 
# conda create -n medasr python=3.10 -y
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate medasr


# pip install uv
# # install dependencies
# uv pip install git+https://github.com/huggingface/transformers.git@65dc261512cbdb1ee72b88ae5b222f2605aad8e5

# # omilinugal asr setup
# conda create -n omilinugal python=3.10 -y
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate omilinugal

# pip install uv

# pip install omnilingual-asr

# gemma model
conda create -n gemma python=3.10 -y
source ~/anaconda3/etc/profile.d/conda.sh
conda activate gemma
pip install -r ../requirements/requirements_gemma4.txt

# #  hugging face models
# conda create -n hf python=3.10 -y
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate hf
# pip install -U transformers torch accelerate


# conda create -n evaluation python=3.10 -y
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate evaluation
# pip install unbabel-comet
# pip install jiwer
# pip install numpy
# pip install pandas
# pip install openai
# pip install whisper_normalizer



