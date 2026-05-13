In muse_bench:
 
1. Make a conda env
conda env create -f environment.yml
conda activate py310

2. Load data from HuggingFace
# Make a huggingface account and get a token
huggingface-cli login
python load_data.py

3. Run finetuning
python finetune.py

3. Run unlearning
cd baselines
bash ./scripts/unlearn_books.sh

4. Run evaluation
python eval.py --model_dirs "baselines/ckpt/books/ga/checkpoint-128" --names "smollm_ga" --corpus books --out_file "out.csv" --tokenizer_dir "HuggingFaceTB/SmolLM-135M"