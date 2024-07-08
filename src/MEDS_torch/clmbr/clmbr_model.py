import os
import shutil

import femr.models.config as femr_config
import femr.models.processor as femr_processor
import femr.models.tasks as femr_tasks
import femr.models.tokenizer as femr_tokenizer
import femr.models.transformer as femr_transformer
import transformers
from datasets import Dataset


class CLMBR:
    def __init__(self, target_dir, num_proc=4, vocab_size=128, tokens_per_batch=32, num_train_epochs=20):
        """Initialize the CLMBR model class with the specified configuration parameters.

        Parameters:
            target_dir (str): The directory where model data will be stored. It ensures that all model-related
                            files are organized in a specific directory.
            num_proc (int): Number of processes to use for parallel operations like data loading and
                            preprocessing.
            vocab_size (int): The size of the vocabulary for the tokenizer, defining how many unique
                            tokens/codes the tokenizer can handle.
            tokens_per_batch (int): Number of tokens per batch during training, which defines the batch size
                                    based on the number of tokens rather than the number of sequences.
            num_train_epochs (int): The number of epochs for which the model will be trained.
        """
        self.target_dir = target_dir
        self.num_proc = num_proc
        self.vocab_size = vocab_size
        self.tokens_per_batch = tokens_per_batch
        self.num_train_epochs = num_train_epochs

        # Ensure the target directory is clean and exists
        if os.path.exists(self.target_dir):
            shutil.rmtree(self.target_dir)
        os.makedirs(self.target_dir, exist_ok=True)
        os.makedirs(os.path.join(self.target_dir, "clmbr_model"), exist_ok=True)

    def load_datasets(self, main_path):
        """Load datasets from a main directory that contains subdirectories for training, held-out, and tuning
        sets.

        Args:
            main_path (str): Path to the main directory containing subdirectories for the datasets.

        Returns:
            dict: A dictionary containing the training, held-out, and tuning datasets.
        """
        # Construct paths to each dataset based on the main directory
        train_path = os.path.join(main_path, "train")
        held_out_path = os.path.join(main_path, "held_out")
        tuning_path = os.path.join(main_path, "tuning")

        # Load each dataset from its respective subdirectory
        train_dataset = Dataset.from_parquet(train_path)
        held_out_dataset = Dataset.from_parquet(held_out_path)
        tuning_dataset = Dataset.from_parquet(tuning_path)

        # Create a dictionary to hold the datasets
        datasets = {"train": train_dataset, "held_out": held_out_dataset, "tuning": tuning_dataset}
        return datasets

    def train_tokenizer(self, datasets):
        """Train a tokenizer using the training dataset.

        Args:
            datasets (dict): A dictionary containing 'train', 'held_out', and 'tuning' datasets.

        Returns:
            tuple: A tuple containing the transformer configuration and the trained tokenizer.
        """
        tokenizer = femr_tokenizer.train_tokenizer(
            datasets["train"], vocab_size=self.vocab_size, num_proc=self.num_proc
        )
        tokenizer.save_pretrained(os.path.join(self.target_dir, "clmbr_model"))

        transformer_config = femr_config.FEMRTransformerConfig(
            vocab_size=tokenizer.vocab_size,
            is_hierarchical=tokenizer.is_hierarchical,
            n_layers=2,
            hidden_size=64,
            intermediate_size=128,
            n_heads=8,
        )
        return transformer_config, tokenizer

    def create_batches(self, tokenizer, datasets):
        """Create batches from the datasets using a specified tokenizer.

        Args:
            tokenizer: The tokenizer trained on the training dataset.
            datasets (dict): A dictionary containing 'train', 'held_out', and 'tuning' datasets.

        Returns:
            tuple: A tuple containing the processed train batches, batch processor, and CLMBR task.
        """
        clmbr_task = femr_tasks.CLMBRTask(clmbr_vocab_size=64)  # TODO number of codes
        processor = femr_processor.FEMRBatchProcessor(tokenizer, clmbr_task)
        train_batches = processor.convert_dataset(
            datasets["train"], tokens_per_batch=self.tokens_per_batch, num_proc=self.num_proc
        )
        val_batches = processor.convert_dataset(
            datasets["tuning"], tokens_per_batch=self.tokens_per_batch, num_proc=self.num_proc
        )
        train_batches.set_format("pt")
        val_batches.set_format("pt")
        return train_batches, val_batches, processor, clmbr_task

    def train_model(self, transformer_cfg, clmbr_task, processor, train_batches, val_batches):
        """Train the CLMBR model using the specified configuration and batches.

        Args:
            transformer_cfg: Configuration for the transformer model.
            clmbr_task: The CLMBR task associated with the model.
            processor: The batch processor used to process datasets.
            train_batches: The batches of training data.

        Effects:
            Trains the model and saves it to the specified directory.
        """
        cfg = femr_config.FEMRModelConfig.from_transformer_task_configs(
            transformer_cfg, clmbr_task.get_task_config()
        )
        model = femr_transformer.FEMRModel(cfg)
        trainer_cfg = transformers.TrainingArguments(
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            output_dir=os.path.join(self.target_dir, "tmp_trainer"),
            remove_unused_columns=False,
            num_train_epochs=self.num_train_epochs,
            eval_steps=20,
            evaluation_strategy="steps",
            logging_steps=20,
            logging_strategy="steps",
            prediction_loss_only=True,
        )
        trainer = transformers.Trainer(
            model=model,
            data_collator=processor.collate,
            train_dataset=train_batches,
            eval_dataset=val_batches,
            args=trainer_cfg,
        )
        trainer.train()
        model.save_pretrained(os.path.join(self.target_dir, "clmbr_model"))

    def evaluate_model(self, model, processor, datasets):
        """Evaluate the model using a specified dataset.

        Args:
            model: The trained model object.
            processor: The batch processor used to process datasets.
            datasets (dict): A dictionary containing 'train', 'held_out', and 'tuning' datasets.

        Returns:
            dict: A dictionary containing evaluation metrics such as loss and accuracy.
        """
        # Process the dataset to create batches in the required format
        eval_batches = processor.convert_dataset(
            datasets["held_out"], tokens_per_batch=self.tokens_per_batch, num_proc=self.num_proc
        )
        eval_batches.set_format("pt")

        # Perform evaluation using the transformers.Trainer object
        results = model.evaluate(eval_batches)
        return results


# Example usage:
# clmbr = CLMBR(target_dir='/path/to/target/dir')
# train_dataset = clmbr.load_datasets('/path/to/parquet/data')
# transformer_cfg, tokenizer = clmbr.train_tokenizer(train_dataset)
# train_batches, processor, clmbr_task = clmbr.create_batches(tokenizer, train_dataset)
# clmbr.train_model(transformer_cfg, clmbr_task, processor, train_batches)
# evaluation_results = clmbr.evaluate_model(model, processor, datasets['held_out'])
# print("Evaluation Results:", evaluation_results)
