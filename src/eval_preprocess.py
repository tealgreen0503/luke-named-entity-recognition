from typing import TypedDict

import numpy as np
from datasets import Dataset
from seqeval.metrics import accuracy_score, f1_score

from src.constant import NON_ENTITY, EntitySpan


def get_metrics_function(dataset: Dataset, id2label: dict[int, str]):
    def compute_metrics(eval_pred):
        batch_probabilities, _ = eval_pred
        batch_id, batch_text, batch_entity_spans, batch_iob2_labels = (
            dataset[column] for column in ["id", "text", "entity_spans", "iob2_labels"]
        )
        batch_text, batch_entity_spans, batch_iob2_labels, batch_probabilities = aggregate_entity_spans(
            batch_id, batch_text, batch_entity_spans, batch_iob2_labels, batch_probabilities
        )
        iob2_predictions = batch_convert_probabilities_to_iob2_predictions(
            batch_text, batch_entity_spans, batch_probabilities, id2label
        )
        iob2_labels = batch_iob2_labels

        return {
            "f1": f1_score(iob2_labels, iob2_predictions, average="micro"),
            "accuracy": accuracy_score(iob2_labels, iob2_predictions),
        }

    return compute_metrics


def aggregate_entity_spans(
    batch_id: list[int],
    batch_text: list[str],
    batch_entity_spans: list[list[EntitySpan]],
    batch_iob2_labels: list[list[str]],
    batch_probabilities: list[np.ndarray],
) -> tuple[list[str], list[list[EntitySpan]], list[list[str]], list[np.ndarray]]:
    """複数のデータにまたがる同一テキストに対するエンティティスパンを統合する。"""

    class AggregatedSample(TypedDict):
        text: str
        entity_spans: list[EntitySpan]
        iob2_labels: list[str]
        probabilities: list[np.ndarray]

    aggregated_samples: dict[int, AggregatedSample] = {}
    for id, text, entity_spans, iob2_labels, probabilities in zip(
        batch_id, batch_text, batch_entity_spans, batch_iob2_labels, batch_probabilities, strict=True
    ):
        if id not in aggregated_samples:
            aggregated_samples[id] = {
                "text": text,
                "entity_spans": entity_spans,
                "iob2_labels": iob2_labels,
                "probabilities": [probabilities],
            }
        else:
            aggregated_samples[id]["entity_spans"] += entity_spans
            aggregated_samples[id]["probabilities"].append(probabilities)

    batch_text = []
    batch_entity_spans = []
    batch_iob2_labels = []
    batch_probabilities = []
    for value in aggregated_samples.values():
        batch_text.append(value["text"])
        batch_entity_spans.append(value["entity_spans"])
        batch_iob2_labels.append(value["iob2_labels"])
        probabilities = np.concatenate(value["probabilities"], axis=0)
        probabilities = probabilities[: len(value["entity_spans"])]
        batch_probabilities.append(probabilities)

    return batch_text, batch_entity_spans, batch_iob2_labels, batch_probabilities


def batch_convert_probabilities_to_iob2_predictions(
    batch_text: list[str],
    batch_entity_spans: list[list[EntitySpan]],
    batch_probabilities: np.ndarray,
    id2label: dict[int, str],
) -> list[list[str]]:
    return [
        convert_probabilities_to_iob2_predictions(text, entity_spans, probabilities, id2label)
        for text, entity_spans, probabilities in zip(batch_text, batch_entity_spans, batch_probabilities, strict=True)
    ]


def convert_probabilities_to_iob2_predictions(
    text: str,
    entity_spans: list[EntitySpan],
    probabilities: np.ndarray,
    id2label: dict[int, str],
) -> list[str]:
    """予測確率を一文字単位のIOB2形式のラベルに変換する。"""

    prediction_labels = np.argmax(probabilities, axis=1)
    prediction_probabilities = np.max(probabilities, axis=1)

    predictions = []
    for span, label, probability in zip(entity_spans, prediction_labels, prediction_probabilities, strict=True):
        if label != 0:
            predictions.append(
                {
                    "start": span[0],
                    "end": span[1],
                    "label": label,
                    "probability": probability,
                }
            )

    iob2_predictions = [NON_ENTITY] * len(text)
    # 確率の高い順から一文字単位で予測結果を反映していく
    for prediction in sorted(predictions, key=lambda pred: pred["probability"], reverse=True):
        start_position = prediction["start"]
        end_position = prediction["end"]
        label = id2label[prediction["label"]]
        if all([tag == NON_ENTITY for tag in iob2_predictions[start_position:end_position]]):
            iob2_predictions[start_position] = f"B-{label}"
            span_length = end_position - start_position
            if span_length >= 2:
                iob2_predictions[start_position + 1 : end_position] = [f"I-{label}"] * (span_length - 1)

    return iob2_predictions
