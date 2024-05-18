from transformers import DistilBertForTokenClassification, Trainer, TrainingArguments
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import TrainerCallback
from copy import deepcopy

from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score

import os

from ner_dataloader import NERDataLoader
from log_helper import save_logs

labels = ["O", "B-CROP", "I-CROP", "B-PEST", "I-PEST", "B-SEED", "I-SEED"]
label2id = {label: idx for idx, label in enumerate(labels)}
id2label = {idx: label for label, idx in label2id.items()}

training_args = TrainingArguments(
    output_dir='../output_model',
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    evaluation_strategy="epoch",  # Evaluate at the end of each epoch
    save_total_limit=1,
    dataloader_pin_memory=False
)


class CustomCallback(TrainerCallback):
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy
        

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    return {
        "accuracy_score": accuracy_score(true_labels, true_predictions),
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
    }


def get_file_path():
    # Get the path to the data folder
    data_folder = "../data"

    # read the file names in the folder
    file_names = os.listdir(data_folder)

    # Get the path to the file
    file_path = os.path.join(data_folder, file_names[0])

    return file_path


if __name__ == '__main__':
    train_dataset = NERDataLoader('../data/train.csv').get_ner_dataset()
    test_dataset = NERDataLoader('../data/test.csv').get_ner_dataset()

    model = DistilBertForTokenClassification.from_pretrained('distilbert-base-uncased', num_labels=len(labels))
    model.config.id2label = id2label
    model.config.label2id = label2id

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    trainer.add_callback(CustomCallback(trainer))
    trainer.train()

    save_logs(trainer.state.log_history)