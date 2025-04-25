import os
import tempfile
import joblib
import s3fs
import kagglehub
import pandas as pd
import json
from datetime import datetime
from datasets import (
    load_dataset,
    load_from_disk,
    DatasetDict,
    concatenate_datasets,
    Dataset
)
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer
)
from src.my_log import get_logger


logger = get_logger(name=__name__)


class CausalLMTrainer(Trainer):
    """
    A custom Trainer class for causal language modeling.
    This trainer ensures that the `input_ids` are used as labels during training,
    which is required for autoregressive (causal) language modeling.
    """
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Ensure we set up the labels correctly for causal language modeling
        """
        labels = inputs["input_ids"].clone()            # Set labels as input_ids for causal language modeling
        inputs["labels"] = labels
        outputs = model(**inputs)
        loss = outputs.loss                             # Get the loss from model outputs
        return (loss, outputs) if return_outputs else loss


class TrainingLLM:
    """
    A class to train the model from retrieving the dataset to fine-tune gpt-2
    """
    poem_type: str
    """
    The type of poem to train on : haiku or classic
    """
    s3_uri: str
    """
    The base URI to the S3 bucket used to save and load models and datasets
    """
    model: GPT2LMHeadModel
    """
    The GPT-2 model to be fine-tuned
    """
    tokenizer: GPT2Tokenizer
    """
    The tokenizer corresponding to the GPT-2 model
    """

    def __init__(self, poem_type, s3_uri=None):
        self.poem_type = poem_type
        self.s3_uri = s3_uri.rstrip("/") if s3_uri else None
        self.fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": "https://minio.lab.sspcloud.fr"})
        self.model, self.tokenizer = self._load_gpt2()

        # add token padding if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self.model.resize_token_embeddings(len(self.tokenizer), mean_resizing=False)

    def _s3_exists(self, path):
        return self.fs.exists(path)

    def _load_from_s3(self, path):
        with self.fs.open(path, "rb") as f:
            return joblib.load(f)

    def _save_to_s3(self, obj, path):
        with self.fs.open(path, "wb") as f:
            joblib.dump(obj, f)

    def _load_gpt2(self):
        gpt2_s3_path = f"{self.s3_uri}/Phoetry/Pretrained_model/gpt2"

        if self._s3_exists(gpt2_s3_path):
            logger.info(f"Loading model and tokenizer from {gpt2_s3_path} on S3...")

            with tempfile.TemporaryDirectory() as tmpdir:
                files = self.fs.ls(gpt2_s3_path, detail=False)
                for file_path in files:
                    filename = os.path.basename(file_path)
                    local_path = os.path.join(tmpdir, filename)
                    self.fs.get(file_path, local_path)

                model = GPT2LMHeadModel.from_pretrained(tmpdir)
                tokenizer = GPT2Tokenizer.from_pretrained(tmpdir)

        else:
            logger.info("Model and tokenizer not found on S3. Downloading from HuggingFace...")
            model = GPT2LMHeadModel.from_pretrained("gpt2")
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    model.save_pretrained(tmpdir)
                    tokenizer.save_pretrained(tmpdir)

                    for file in os.listdir(tmpdir):
                        local_path = os.path.join(tmpdir, file)
                        remote_path = f"{gpt2_s3_path}/{file}"
                        self.fs.put(local_path, remote_path)

                    logger.info(f"Model and tokenizer uploaded to {gpt2_s3_path} on S3.")
            except Exception as e:
                logger.error(f"Error saving model and tokenizer to S3: {e}")

        return model, tokenizer

    def retrieve_dataset(self):
        dataset_s3_path = f"{self.s3_uri}/Phoetry/Datasets/{self.poem_type}"

        if self._s3_exists(dataset_s3_path):
            logger.info("Loading dataset from S3 via save_to_disk/load_from_disk...")
            with tempfile.TemporaryDirectory() as tmpdir:
                self.fs.get(dataset_s3_path, tmpdir, recursive=True)
                local_dataset_dir = os.path.join(tmpdir, self.poem_type)
                return load_from_disk(local_dataset_dir)

        logger.info("Dataset not found on S3. Downloading and preparing locally...")
        if self.poem_type == "haiku":
            logger.info("Loading haiku dataset...")
            dataset = load_dataset("statworx/haiku")
        else:
            logger.info("Loading classic poems datasets...")
            dataset = load_dataset("shahules786/PoetryFoundationData")

        logger.info("Saving dataset to S3 via save_to_disk...")
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset.save_to_disk(tmpdir)
            self.fs.put(tmpdir, dataset_s3_path, recursive=True)
        logger.info("Dataset saved to S3.")

        return dataset

    def tokenize_dataset(self, dataset):
        poem_column = "text" if self.poem_type == "haiku" else "content"

        def tokenize_function(batch):
            return self.tokenizer(batch[poem_column], truncation=True, padding="max_length", max_length=512)

        logger.info("Tokenizing dataset...")
        return dataset["train"].map(tokenize_function, batched=True)

    def training_preparation(self):
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            return_tensors="pt"
        )
        training_args = TrainingArguments(
            output_dir=f"{self.s3_uri}/Phoetry/tmp/{self.poem_type}",   # output directory for model checkpoints
            do_eval=False,                                              # evaluate every few steps
            learning_rate=5e-5,                                         # learning rate for optimizer
            per_device_train_batch_size=2,                              # batch size for training
            num_train_epochs=2,                                         # number of training epochs
            save_steps=5000,                                            # save checkpoints every 10,000 steps
            save_total_limit=2,                                         # only keep the 2 most recent checkpoints
            logging_dir=f"{self.s3_uri}/Phoetry/logs/{self.poem_type}", # directory to save logs
            logging_steps=500,                                          # log every 500 steps
            report_to=None,
            no_cuda=False,                                              # If False, forces GPU usage (set True if you want CPU)
            fp16=True                                                   # Use mixed precision for speedup (if using GPU)
        )

        return training_args, data_collator

    def train(self, tokenized_dataset):
        training_args, data_collator = self.training_preparation()
        trainer = CausalLMTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator
        )
        logger.info("Starting training...")
        trainer.train()
        logger.info("Training complete.")

    def save_model(self):
        model_name = f"gpt2_en_{self.poem_type}"
        model_s3_dir = f"{self.s3_uri}/Phoetry/Poem_models/{model_name}"
        base_dir = os.path.dirname(model_s3_dir)

        logger.info(f"Saving model to {model_s3_dir} on S3...")

        files = []

        with tempfile.TemporaryDirectory() as tmpdir:
            self.model.save_pretrained(tmpdir)
            self.tokenizer.save_pretrained(tmpdir)

            for file in os.listdir(tmpdir):
                local_path = os.path.join(tmpdir, file)
                remote_path = f"{model_s3_dir}/{file}"
                self.fs.put(local_path, remote_path)
                files.append(file)
                logger.debug(f"Uploaded {local_path} to {remote_path}")

            logger.info(f"Model and tokenizer successfully uploaded to {model_s3_dir}")

            metadata = {
                "name": model_name,
                "description": f"GPT-2 model fine-tuned for generating {self.poem_type} poems",
                "version": "1.0",
                "status": "trained",
                "date_of_release": datetime.utcnow().isoformat(),
                "URL": base_dir,
                "files": files
            }

            json_path = os.path.join(tmpdir, f"{self.poem_type}_metadata.json")
            with open(json_path, "w") as f:
                json.dump(metadata, f, indent=4)

            metadata_s3_path = f"{model_s3_dir}/metadata.json"
            self.fs.put(json_path, metadata_s3_path)
            logger.info(f"Metadata JSON saved to {metadata_s3_path}")
        try:
            logger.info(f"Scanning model folders in {base_dir}...")

            model_folders = self.fs.ls(base_dir, detail=False)
            model_entries = {}

            for folder_path in model_folders:
                folder_name = os.path.basename(folder_path)
                metadata_path = f"{folder_path}/metadata.json"
                if self.fs.exists(metadata_path):
                    model_entries[folder_name] = metadata_path.replace(
                        "s3://", "https://minio.lab.sspcloud.fr/"
                    )

            with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as tmpfile:
                json.dump(model_entries, tmpfile, indent=4)
                tmpfile_path = tmpfile.name

            registry_path = f"{base_dir}/models_available.json"
            self.fs.put(tmpfile_path, registry_path)
            logger.info(f"models_available.json updated at {registry_path}")

        except Exception as e:
            logger.warning(f"Failed to update models_available.json: {e}")
