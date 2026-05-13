from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling

forget_train = load_dataset("muse-bench/MUSE-Books", "train", split="forget")
retain_train = load_dataset("muse-bench/MUSE-Books", "train", split="retain2")

train_data = concatenate_datasets([forget_train, retain_train])

model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M")
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")

tokenizer.pad_token = tokenizer.eos_token

def tokenize(example):
    return tokenizer(example["text"], truncation=True, max_length=512)

tokenized = train_data.map(tokenize, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./muse_finetuned",
    num_train_epochs=5,
    learning_rate=1e-5,
    per_device_train_batch_size=32,
    save_strategy="epoch",
    fp16=True,
    save_safetensors=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=data_collator
)

trainer.train()
trainer.save_model("./muse_finetuned/final")