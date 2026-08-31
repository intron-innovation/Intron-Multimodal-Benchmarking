# activate evaluation environment and run evaluation script
source ~/anaconda3/etc/profile.d/conda.sh
conda activate evaluation

# automatic metrics: WER/CER, BLEU/chrF/COMET, and Spoken QA COMET
python scripts/evaluations.py

# Spoken QA rubric: per-dimension tables from the expert-panel ratings, plus
# the panel's inter-rater reliability. Add --judge-scores to also report how
# closely an LLM judge (scripts/qa_rubric_judge.py) tracks the panel.
HUMAN_RATINGS="data/Spoken QA/expert_panel_ratings.csv"
if [ -f "$HUMAN_RATINGS" ]; then
    python scripts/qa_rubric_evals.py --human "$HUMAN_RATINGS" --modality audio
else
    echo "Skipping Spoken QA rubric evaluation: $HUMAN_RATINGS not found."
    echo "See data/Spoken QA/samples/ for the expected format."
fi
