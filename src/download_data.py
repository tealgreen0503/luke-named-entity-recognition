import argparse
import json
import os
import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def download_data(valid_size: float = 0.1, test_size: float = 0.2, seed: int = 42) -> None:
    data_url = "https://github.com/stockmarkteam/ner-wikipedia-dataset/raw/main/ner.json"
    data_path = Path("data")
    train_path = data_path / "train.json"
    valid_path = data_path / "valid.json"
    test_path = data_path / "test.json"
    assert valid_size < 1.0 and test_size < 1.0

    raw_data = urllib.request.urlopen(data_url)
    raw_data_list = json.loads(raw_data.read().decode("utf-8"))

    data_list = []
    for id, raw_data in enumerate(raw_data_list):
        label_spans = []
        for entity in raw_data["entities"]:
            label_spans.append(
                {"start": entity["span"][0], "end": entity["span"][1], "word": entity["name"], "label": entity["type"]}
            )
        data_list.append(
            {"id": id, "text": raw_data["text"], "label_spans": label_spans, "is_labeled": len(label_spans) > 0}
        )
    df = pd.DataFrame(data_list)

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, shuffle=True, stratify=df["is_labeled"]
    )
    train_df, valid_df = train_test_split(
        train_df, test_size=valid_size, random_state=seed, shuffle=True, stratify=train_df["is_labeled"]
    )
    os.makedirs(data_path, exist_ok=True)
    train_df.drop(columns="is_labeled").to_json(train_path, orient="records", force_ascii=False, indent=2)
    valid_df.drop(columns="is_labeled").to_json(valid_path, orient="records", force_ascii=False, indent=2)
    test_df.drop(columns="is_labeled").to_json(test_path, orient="records", force_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="data/ner.json")
    parser.add_argument("--valid_size", type=float, default=0.1)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    download_data(args.valid_size, args.test_size, args.seed)
