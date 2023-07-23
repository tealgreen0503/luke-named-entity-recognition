import os
import shutil
import warnings
from pathlib import Path

import pandas as pd
import torch
import yaml
from dotenv import load_dotenv
from transformers import (
    AutoTokenizer,
    LukeForEntitySpanClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.data_collator import DataCollatorForEntitySpanClassification
from src.data_preprocess import create_dataset
from src.eval_preprocess import get_metrics_function
from src.word_tokenizer import AutoWordTokenizer


def main():
    warnings.filterwarnings("ignore")
    load_dotenv()
    with open(Path(os.path.dirname(__file__)) / "config.yaml") as f:
        config = yaml.safe_load(f)
    set_seed(config["seed"])

    id2label = config["id2label"]
    label2id = {label: i for i, label in id2label.items()}

    train_df = pd.read_json(config["path"]["train"])
    valid_df = pd.read_json(config["path"]["valid"])
    test_df = pd.read_json(config["path"]["test"])

    tokenizer = AutoTokenizer.from_pretrained(**config["model_info"], **config["tokenizer_kwargs"], use_fast=True)
    word_tokenizer = AutoWordTokenizer.from_config(**config["word_tokenizer_kwargs"])

    train_dataset = create_dataset(train_df, word_tokenizer, tokenizer, label2id, config["tokenization_kwargs"])
    valid_dataset = create_dataset(valid_df, word_tokenizer, tokenizer, label2id, config["tokenization_kwargs"])
    test_dataset = create_dataset(test_df, word_tokenizer, tokenizer, label2id, config["tokenization_kwargs"])

    data_collator = DataCollatorForEntitySpanClassification(
        tokenizer=tokenizer,
        padding=True,
        max_length=config["tokenization_kwargs"]["max_length"],
        max_entity_length=config["tokenizer_kwargs"]["max_entity_length"],
    )

    model = LukeForEntitySpanClassification.from_pretrained(
        **config["model_info"],
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        **config["model_kwargs"],
    )

    training_args = TrainingArguments(
        **config["trainer"],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        compute_metrics=get_metrics_function(valid_dataset, id2label),
        preprocess_logits_for_metrics=lambda logits, labels: torch.softmax(logits, dim=2),
    )

    trainer.train()
    shutil.rmtree(config["trainer"]["output_dir"])

    eval_metrics = trainer.evaluate()
    print(pd.DataFrame(eval_metrics, index=[0]))

    trainer.compute_metrics = get_metrics_function(test_dataset, id2label)
    test_metrics = trainer.predict(test_dataset).metrics
    print(pd.DataFrame(test_metrics, index=[0]))

    output_dir = Path(config["path"]["fine_tuned_models"]) / config["run_name"]
    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)


if __name__ == "__main__":
    main()
