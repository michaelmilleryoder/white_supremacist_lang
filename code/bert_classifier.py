""" BERT for text classification using HuggingFace Trainer 
    @author Michael Miller Yoder
    @year 2022
"""

import pandas as pd
from datasets import Dataset, load_metric
from transformers import (AutoTokenizer, DataCollatorWithPadding, 
        AutoModelForSequenceClassification, TrainingArguments, Trainer)
import numpy as np


class BertClassifier:

    def __init__(self, load=None):
        """ Args:
                load: None to train a new model from scratch, or a path to the model to load and further train
        """
        if load is None:
            self.model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(load)
            self.tokenizer = AutoTokenizer.from_pretrained(load)
        self.n_epochs = 3
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.metrics = {'accuracy': load_metric('accuracy'), 
                'precision': load_metric('precision'),
                'recall': load_metric('recall'),
               'f1': load_metric('f1')}
        self.batch_size = 16
        self.checkpoint = self.batch_size * int(2e3)
        self.training_args = TrainingArguments(
            logging_dir='../logs',
            output_dir="../models/bert",
            learning_rate=2e-5,
            per_device_train_batch_size = self.batch_size,
            per_device_eval_batch_size = self.batch_size,
            num_train_epochs=3,
            weight_decay=0.01,
            evaluation_strategy='steps',
            save_strategy='steps',
            logging_steps = self.checkpoint,
            eval_steps = self.checkpoint,
            save_steps = self.checkpoint,
        )
        self.train_data = None
        self.train_tokenized = None
        #self.trainer = None
        self.trainer = Trainer(
            model = self.model,
            args = self.training_args,
            tokenizer = self.tokenizer,
            data_collator = self.data_collator,
            compute_metrics = self.compute_metrics,
        )
        self.test_data = None
        self.test_tokenized = None

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {metric_name: metric.compute(
                predictions=predictions, references=labels) for metric_name, metric in self.metrics.items()}

    def train(self, data: pd.DataFrame):
        """ Train the classifier.
            Args:
                data: the training data
        """
        self.train_data, self.train_tokenized = self.prepare_dataset(data, split=True)
        self.trainer.train_dataset = self.train_tokenized["train"],
        self.trainer.eval_dataset = self.train_tokenized["test"],
        #self.trainer = Trainer(
        #    model = self.model,
        #    args = self.training_args,
        #    train_dataset = self.train_tokenized["train"],
        #    eval_dataset = self.train_tokenized["test"],
        #    tokenizer = self.tokenizer,
        #    data_collator = self.data_collator,
        #    compute_metrics = self.compute_metrics,
        #)
        print('Training model...')
        res = self.trainer.train()
        self.trainer.save_metrics('all', res.metrics)

    def prepare_dataset(self, data: pd.DataFrame, split=False):
        """ Prepare dataset into a format for training or evaluating.
            Args:
                data: the training data
            Returns:
                Tuple with (HuggingFace DatasetDict with train and test splits (optionally), tokenized dataset)
        """
        print('Preparing training data...')
        ds = Dataset.from_pandas(data)
        if split:
            ds = ds.train_test_split(test_size=0.1)
        tokenized = ds.map(self.preprocess, batched=True)
        return (ds, tokenized)

    def preprocess(self, examples):
        """ Preprocess HuggingFace dataset """
        return self.tokenizer(examples["text"], truncation=True)

    def evaluate(self, test: pd.DataFrame):
        """ Evaluate the model on an unseen dataset.
            Args:
                test: pandas DataFrame of unseen data, containing 'text' and 'label' columns
        """
        print("Evaluating on test corpus...")
        self.test_data, self.test_tokenized = self.prepare_dataset(test, split=False)
        self.trainer.evaluate(self.test_tokenized)
