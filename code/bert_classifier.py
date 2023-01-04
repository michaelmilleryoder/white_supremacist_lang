""" BERT for text classification using HuggingFace Trainer 
    @author Michael Miller Yoder
    @year 2022
"""
import os
import pdb
from pprint import pprint

import pandas as pd
from datasets import Dataset, DatasetDict, load_metric
from transformers import (AutoTokenizer, DataCollatorWithPadding, 
        AutoModelForSequenceClassification, TrainingArguments, Trainer)
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import GroupShuffleSplit
import scipy

from corpus import Corpus


class WeightedLossTrainer(Trainer):
    """ Custom class to use a loss weighted by labels """
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (with 3 labels
        loss_fct = nn.CrossEntropyLoss(weight=torch.Tensor([1.0, 1.0, 3.0]).cuda())
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


class MultiClassTrainer(Trainer):
    """ Custom class to handle multiclass classification training (NOT USED) """
    
    def compute_loss(self, model, inputs, return_outputs=False):
        #labels = inputs.get('labels')
        labels = inputs.pop('labels')
        #outputs = model(**inputs, labels=labels)
        outputs = model(**inputs)
        logits = outputs[0]
        #loss = outputs.loss
        loss = torch.nn.functional.cross_entropy(logits, labels)
        #loss = torch.nn.BCEWithLogitsLoss(logits, labels)
        #loss = torch.nn.CrossEntropyLoss(logits, labels)
        return (loss, outputs) if return_outputs else loss


class BertClassifier:

    def __init__(self, exp_name: str, train=False, load=None, train_length=None, n_labels: int = 2, 
        id2label: dict = None, label2id: dict = None, pretrained_model: str = 'distilbert-base-uncased', 
        n_epochs: int = 5, checkpoints: str = 'epoch', test_label_combine: dict = None):
        """ Args:
                exp_name: name of the experiment (for the output filename)
                train: whether the model will be trained
                load: None to train a new model from scratch, or a path to the model to load
                train_length: If not None, the length of the training data set, 
                    used to calculate the number of steps before logging and evaluation
                n_labels: number of labels to be used in classification
                id2label: mapping of class IDs to class names
                label2id: mapping of class names to class IDs
                pretrained_model: Hugging Face name of the pretrained model to load and fine-tune (default distilbert-base-uncased)
                n_epochs: number of epochs to train
                checkpoints: whether to save at 'epoch' or 'steps', which will save a fixed number of times over training
                test_label_combine: a dictionary of any changes of predicted labels to make 
                    (to combine 3-way to 2-way classification, eg)
        """
        self.exp_name = exp_name
        self.id2label = id2label
        self.label2id = label2id
        self.pretrained_model = pretrained_model
        if load is None:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.pretrained_model, num_labels=n_labels,
                id2label=self.id2label, label2id=self.label2id)
            self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(load)
            self.tokenizer = AutoTokenizer.from_pretrained(load)
        self.n_epochs = n_epochs
        self.checkpoints = checkpoints if checkpoints in ['steps', 'epoch'] else 'no'
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.metrics = {'accuracy': load_metric('accuracy'), 
                'precision': load_metric('precision'),
                'recall': load_metric('recall'),
               'f1': load_metric('f1')
            }
        self.test_label_combine = test_label_combine
        #if n_labels == 2:
        #    self.metrics = {'accuracy': load_metric('accuracy'), 
        #            'precision': load_metric('precision'),
        #            'recall': load_metric('recall'),
        #           'f1': load_metric('f1')}
        #else:
        #    self.metrics = {'accuracy': load_metric('accuracy'), 
        #            'precision': load_metric('precision', average=None),
        #            'recall': load_metric('recall', average=None),
        #           'f1': load_metric('f1', average=None)}
        self.batch_size = 16
        if train_length is None:
            self.checkpoint_steps = self.batch_size * int(2e3)
        else:
            total_steps = (int(train_length/self.batch_size) + 1) * self.n_epochs
            self.checkpoint_steps = int(total_steps/30)
            #pdb.set_trace() # check that train_length, steps is working as expected (doesn't match progress bar for some reason)
            #    # Probably is batch size "per device" * devices or something? Can't check DataLoader
        self.output_dir = f"../output/bert/{self.exp_name}"
        if train:
            report_to = 'wandb'
            run_name = self.exp_name
        else:
            report_to = None
            run_name = None
            os.environ['WANDB_DISABLED'] = 'True'
        self.training_args = TrainingArguments(
            logging_dir='../logs',
            output_dir=self.output_dir,
            learning_rate=2e-5,
            per_device_train_batch_size = self.batch_size,
            per_device_eval_batch_size = self.batch_size,
            num_train_epochs=self.n_epochs,
            weight_decay=0.01,
            evaluation_strategy=self.checkpoints,
            save_strategy=self.checkpoints,
            logging_strategy=self.checkpoints,
            logging_steps = self.checkpoint_steps,
            eval_steps = self.checkpoint_steps,
            save_steps = self.checkpoint_steps,
            report_to = report_to,
            run_name=run_name,
        )
        self.train_data = None
        self.train_tokenized = None
        #if n_labels > 2:
        #    self.trainer_class = MultiClassTrainer
        #else:
        #    self.trainer_class = Trainer
        self.trainer_class = Trainer
        #self.trainer_class = WeightedLossTrainer
        if train:
            self.trainer = None
        else:
            self.trainer = self.trainer_class(
                model = self.model,
                args = self.training_args,
                tokenizer = self.tokenizer,
                data_collator = self.data_collator,
                compute_metrics = self.compute_metrics,
            )
        self.test_data = None
        self.test_tokenized = None

    def compute_metrics(self, eval_pred):
        # TODO: Add in evaluation on external corpora, or put it in another callback sort of thing
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        #predictions = self.return_preds(logits) # for shifts
        if self.test_label_combine is not None:
            predictions = np.array([self.label2id[self.test_label_combine.get(self.id2label[pred], 
                self.id2label[pred])] for pred in predictions])
        results = {}
        for metric_name, metric in self.metrics.items():
            if metric_name in ['precision', 'recall', 'f1']:
                unique_labels = sorted(set(predictions.tolist() + labels.tolist()))
                results[metric_name] = {}
                # Get results for each class
                for k, result in metric.compute(predictions=predictions, references=labels, average=None).items():
                    results[metric_name][k] = {}
                    if isinstance(result, float):
                        # Get label ID from predictions
                        label_idx = list(set(predictions))[0]
                        label = self.id2label[label_idx]
                        results[metric_name][k][label] = result
                    else:
                        for i, val in enumerate(result): # result is an array for each class, or just a number if there's one class
                            label_idx = unique_labels[i]
                            if not label_idx in self.id2label:
                                pdb.set_trace()
                            label = self.id2label[label_idx]
                            results[metric_name][k][label] = val
                #results[metric_name] = {k: {self.id2label[unique_labels[i]]: val for i, val in enumerate(result)} for (k, 
                #    result) in metric.compute(predictions=predictions, references=labels, average=None).items()}
                # Weighted F1, prec, recall
                for k, result in metric.compute(predictions=predictions, references=labels, average='weighted').items():
                    results[metric_name][k]['weighted'] = result
            else:
                results[metric_name] = metric.compute(predictions=predictions, references=labels)
        return results

    def return_preds(self, logits):
        """ Return predictions based on customized decision thresholds for classes """
        white_supremacist_idx = self.label2id['white_supremacist']

        # Modify logits to prioritize higher precision
        ws_shift = -0.5
        modified = logits.copy()
        modified[:,white_supremacist_idx] = modified[:,white_supremacist_idx] + ws_shift
        
        return np.argmax(modified, axis=-1)
        

    def train(self, data: pd.DataFrame):
        """ Train the classifier.
            Args:
                data: the training data
        """
        self.train_data, self.train_tokenized = self.prepare_dataset(data, split=True)
        #self.trainer.train_dataset = self.train_tokenized["train"],
        #self.trainer.eval_dataset = self.train_tokenized["test"],
        # Tried and failed to specify most of the params in init and just specify train and eval datasets here
        self.trainer = self.trainer_class(
            model = self.model,
            args = self.training_args,
            train_dataset = self.train_tokenized["train"],
            eval_dataset = self.train_tokenized["test"],
            tokenizer = self.tokenizer,
            data_collator = self.data_collator,
            compute_metrics = self.compute_metrics,
        )
        print('Training model...')
        res = self.trainer.train()
        self.trainer.save_model() # I assume saves to the output dir
        self.trainer.save_metrics('all', res.metrics)

    def prepare_dataset(self, data: pd.DataFrame, split=False):
        """ Prepare dataset into a format for training or evaluating.
            Args:
                data: the training data
            Returns:
                Tuple with (HuggingFace DatasetDict with train and test splits (optionally), tokenized dataset)
        """
        print('Preparing training data...')
        # TODO: update to split keeping indices independent between train and test with GroupShuffleSplit
        if split:
            ds = DatasetDict()
            splitter = GroupShuffleSplit(test_size=0.1, n_splits=1, random_state=9)
            train_inds, test_inds = next(splitter.split(data, groups=data.index))
            ds['train'] = Dataset.from_pandas(data.iloc[train_inds])
            ds['test'] = Dataset.from_pandas(data.iloc[test_inds])
        else:
            ds = Dataset.from_pandas(data)

        tokenized = ds.map(self.preprocess, batched=True)
        #ds = Dataset.from_pandas(data)
        #if split:
        #    ds = ds.train_test_split(test_size=0.1, seed=9)
        #tokenized = ds.map(self.preprocess, batched=True)
        return (ds, tokenized)

    def preprocess(self, examples):
        """ Preprocess HuggingFace dataset """
        return self.tokenizer(examples["text"], truncation=True)

    def evaluate(self, test: list[Corpus]):
        """ Evaluate the model on unseen corpora. Gives separate evaluations per dataset within the corpora
            Args:
                test: list of Corpus objects with pandas DataFrame of unseen data in corpus.data, 
                    containing 'text' and 'label' columns
        """
        result_lines = []
        for corpus in test:
            print(f"\nEvaluating on test corpus {corpus.name}...")
            for dataset in corpus.data.dataset.unique():
                selected = corpus.data.query('dataset==@dataset')
                test_data, test_tokenized = self.prepare_dataset(selected, split=False)
                res = self.trainer.evaluate(test_tokenized)
                for metric in ['precision', 'recall', 'f1']:
                    for label, value in res[f'eval_{metric}'][metric].items():
                        result_line = {'dataset': dataset,
                            'metric': metric,
                            'label': label,
                            'value': value
                        }
                        result_lines.append(result_line)
                result_lines.append({'dataset': dataset, 'metric': 'accuracy', 'value': res['eval_accuracy']['accuracy']})
                pred_output = self.trainer.predict(test_tokenized)
                preds = np.argmax(pred_output.predictions, axis=-1)
                # Save out numeric class probability predictions
                prob_outpath = os.path.join(self.output_dir, f'{dataset}_pred_probs.txt')
                prob = scipy.special.softmax(pred_output.predictions, axis=-1)
                class_prob = pd.DataFrame(prob)
                class_prob.columns = class_prob.columns.map(self.id2label)
                class_prob.to_json(prob_outpath, orient='records', lines=True)
                # Save out class name predictions
                pred_outpath = os.path.join(self.output_dir, f'{dataset}_predictions.json')
                preds_classnames = pd.Series(preds).map(self.id2label).to_json(pred_outpath)
        results = pd.DataFrame(result_lines)
        outpath = os.path.join(self.output_dir, 'results.jsonl')
        results.to_json(outpath, orient='records', lines=True)
        print(f"Saved results to {outpath}")
