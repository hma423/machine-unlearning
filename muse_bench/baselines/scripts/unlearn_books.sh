CORPUS="books"

FORGET="../data/$CORPUS/raw/forget.txt"
RETAIN="../data/$CORPUS/raw/retain1.txt"

TARGET_DIR="HuggingFaceTB/SmolLM-135M"
LLAMA_DIR="HuggingFaceTB/SmolLM-135M"

MAX_LEN=2048
EPOCHS=1
LR='1e-5'
PER_DEVICE_BATCH_SIZE=4 # 8 GPUs
FT_EPOCHS=1
FT_LR='1e-5'


for algo in 'ga' ; do # 'ga_gdr' 'ga_klr' 'npo' 'npo_gdr' 'npo_klr'
    python unlearn.py \
        --algo $algo \
        --model_dir $TARGET_DIR --tokenizer_dir $LLAMA_DIR \
        --data_file $FORGET --retain_data_file $RETAIN \
        --out_dir "./ckpt/$CORPUS/$algo" \
        --max_len $MAX_LEN --epochs $EPOCHS --lr $LR \
        --per_device_batch_size $PER_DEVICE_BATCH_SIZE
done


# python unlearn.py \
#     --algo 'tv' \
#     --model_dir $TARGET_DIR --tokenizer_dir $LLAMA_DIR \
#     --data_file $FORGET --retain_data_file $RETAIN \
#     --out_dir "./ckpt/$CORPUS/tv" \
#     --max_len $MAX_LEN --epochs $FT_EPOCHS --lr $FT_LR \
#     --per_device_batch_size $PER_DEVICE_BATCH_SIZE \
#     --alpha 5.0
