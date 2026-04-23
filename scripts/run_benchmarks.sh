# parse command line arguments, task
task=$1

# Transcription block
if [ "$task" == "transcription" ] || [ "$task" == "all" ]; then
    echo "Running transcription benchmark..."
    # activate conda environment depending on model
    source ~/anaconda3/etc/profile.d/conda.sh
    # available models: gemma, omilinugal, Intron Sahara, Google Gemini, Open AI, Azure Speech
    echo "Available models: gemma, sahara, sahara, google_gemini, open_ai, azure"
    # loop through models and run benchmark for each model
    for model in medasr ; do
        echo "Running benchmark for model: $model"
        if [ "$model" == "gemma4" ]; then
            conda activate gemma
            echo "Running transcription benchmark for model: $model"
            
        elif [ "$model" == "sahara" ]; then
            conda activate nemo
            echo "Running transcription benchmark for model: $model"

        elif [ "$model" == "medasr" ]; then
            conda activate medasr
            echo "Running transcription benchmark for model: $model"
           
        else
            conda activate hf
          
        fi
        python scripts/transcription_benchmark.py --model "$model"
        
    done
fi

# Translation block
if [ "$task" == "translation" ] || [ "$task" == "all" ]; then
    echo "Running translation benchmark..."
    # activate conda environment depending on model
    source ~/anaconda3/etc/profile.d/conda.sh
    # available models: gemma, omilinugal, Intron Sahara, Google Gemini, Open AI, Azure Speech
    echo "Available models: gemma, omilinugal, sahara, google_gemini, open_ai, azure"
    # loop through models and run benchmark for each model
    for model in gemma4 ; do
        echo "Running benchmark for model: $model"
        if [ "$model" == "gemma4" ]; then
            conda activate gemma
            echo "Running translation benchmark for model: $model"
            
        elif [ "$model" == "omilinugal" ]; then
            conda activate omilinugal
           
        else
            conda activate hf
          
        fi
        python scripts/translation_benchmark.py --model "$model"
        
    done
fi

# QA block
if [ "$task" == "qa" ] || [ "$task" == "all" ]; then
    echo "Running qa benchmark..."
    # activate conda environment depending on model
    source ~/anaconda3/etc/profile.d/conda.sh
    # available models: gemma, omilinugal, Intron Sahara, Google Gemini, Open AI, Azure Speech
    echo "Available models: gemma, omilinugal, sahara, google_gemini, open_ai, azure"
    # loop through models and run benchmark for each model
    for model in gemma4 ; do
        echo "Running benchmark for model: $model"
        if [ "$model" == "gemma4" ]; then
            conda activate gemma
            echo "Running qa benchmark for model: $model"
            
        elif [ "$model" == "omilinugal" ]; then
            conda activate omilinugal
           
        else
            conda activate hf
          
        fi
        python scripts/qa_benchmark.py --model "$model"
        
    done
# Error handling (only triggers if task is none of the above)
elif [ "$task" != "transcription" ] && [ "$task" != "translation" ] && [ "$task" != "qa" ] && [ "$task" != "all" ]; then
    echo "Invalid task. Please specify 'transcription', 'translation', 'qa', or 'all'."
fi