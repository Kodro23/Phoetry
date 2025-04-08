from datasets import load_dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)


# Load dataset
poems = load_dataset("shahules786/PoetryFoundationData")

# Initialize model
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)


# create a tokenize function
def tokenize_function(dataset):
    poem = dataset["content"]
    tokenizer.truncation_side = "left"
    tokenized_inputs = tokenizer(
        poem,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    return tokenized_inputs


# add pad token if none exists
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.resize_token_embeddings(len(tokenizer))


tokenized_poems = poems["train"].map(tokenize_function, batched=True)
tokenized_poems

# Format dataset
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    return_tensors='pt',
    mlm=False
)

# Set up the training arguments
training_args = TrainingArguments(
    output_dir="./trained_model",  # output directory for model checkpoints
    do_eval=False,  # evaluate every few steps
    learning_rate=5e-5,  # learning rate for optimizer
    per_device_train_batch_size=2,  # batch size for training
    num_train_epochs=2,  # number of training epochs
    save_steps=10_000,  # save checkpoints every 10,000 steps
    save_total_limit=5,  # only keep the 2 most recent checkpoints
    logging_dir="./logs",  # directory to save logs
    logging_steps=500,  # log every 500 steps
    report_to=None,
    no_cuda=False,  # If False, forces GPU usage (set True if you want CPU)
    fp16=True,  # Use mixed precision for speedup (if using GPU
)


class CausalLMTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Ensure we set up the labels correctly for causal language modeling
        labels = inputs.get(
            "input_ids"
        ).clone()  # Set labels as input_ids for causal language modeling
        outputs = model(**inputs)
        loss = outputs.loss  # Get the loss from model outputs
        return (loss, outputs) if return_outputs else loss


# Set up the Trainer
trainer = CausalLMTrainer(
    model=model,                         # the model to train
    args=training_args,                  # training arguments
    train_dataset=tokenized_poems,          # training data    
    data_collator=data_collator
)

# Train the model
trainer.train()

# Save model
model.save_pretrained("trained_model/poet-gpt2")
tokenizer.save_pretrained("trained_model/poet-gpt2")
