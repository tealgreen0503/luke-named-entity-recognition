from typing import Any

import pandas as pd
from datasets import Dataset
from transformers import (
    BatchEncoding,
    LukeTokenizer,
    MLukeTokenizer,
)

from src.constant import NON_ENTITY, EntitySpan, LabelSpan, WordSpan
from src.word_tokenizer import WordTokenizer


def create_dataset(
    df: pd.DataFrame,
    word_tokenizer: WordTokenizer,
    tokenizer: LukeTokenizer | MLukeTokenizer,
    label2id: dict[str, int],
    tokenize_kwargs=None,
):
    dataset = Dataset.from_pandas(df)

    dataset = dataset.map(
        batch_prepare_inputs_for_model,
        batched=True,
        remove_columns=dataset.column_names,
        fn_kwargs={
            "word_tokenizer": word_tokenizer,
            "tokenizer": tokenizer,
            "label2id": label2id,
            "tokenize_kwargs": tokenize_kwargs,
        },
    )
    dataset = dataset.map(
        batch_prepare_labels_for_evaluation,
        batched=True,
        remove_columns=["label_spans"],
    )

    return dataset


def batch_prepare_inputs_for_model(
    batch_example: dict[str, list],
    word_tokenizer: WordTokenizer,
    tokenizer: LukeTokenizer | MLukeTokenizer,
    label2id: dict[str, int],
    tokenize_kwargs: dict[str, Any] | None = None,
) -> dict[str, list]:
    batch_id, batch_text, batch_label_spans = (batch_example[column] for column in ["id", "text", "label_spans"])

    batch_word_spans = batch_word_tokenize(batch_text, word_tokenizer, tokenizer)
    batch_entity_spans, batch_id, batch_text, batch_label_spans = batch_create_entity_spans(
        batch_word_spans, tokenizer, batch_id, batch_text, batch_label_spans
    )
    assert len(batch_entity_spans) == len(batch_id) == len(batch_text) == len(batch_label_spans)

    batch_encodings = batch_tokenize(batch_text, batch_entity_spans, tokenizer, tokenize_kwargs)

    batch_labels = batch_align_labels(batch_entity_spans, batch_label_spans, label2id)

    return {
        "id": batch_id,
        "text": batch_text,
        "label_spans": batch_label_spans,
        "entity_spans": batch_entity_spans,
        **batch_encodings,
        "labels": batch_labels,
    }


def batch_word_tokenize(
    batch_text: list[str], word_tokenizer: WordTokenizer, tokenizer: LukeTokenizer | MLukeTokenizer
) -> list[list[WordSpan]]:
    """テキストを任意の方法で分割し、word_spansを作成する"""

    batch_word_spans = [word_tokenizer(text) for text in batch_text]
    batch_word_spans = [
        [word_span for word_span in word_spans if len(tokenizer.tokenize(word_span["word"])) != 0]
        for word_spans in batch_word_spans
    ]
    return batch_word_spans


def batch_create_entity_spans(
    batch_word_spans: list[list[WordSpan]],
    tokenizer: LukeTokenizer | MLukeTokenizer,
    batch_id: list[int],
    batch_text: list[str],
    batch_label_spans: list[list[LabelSpan]],
) -> tuple[list[list[EntitySpan]], list[int], list[str], list[list[LabelSpan]]]:
    batch_entity_spans: list[list[EntitySpan]] = []
    new_batch_id: list[int] = []
    new_batch_text: list[str] = []
    new_batch_label_spans: list[list[LabelSpan]] = []
    for word_spans, id, text, label_spans in zip(
        batch_word_spans, batch_id, batch_text, batch_label_spans, strict=True
    ):
        _batch_entity_spans, _batch_id, _batch_text, _batch_label_spans = create_entity_spans(
            word_spans, tokenizer, id, text, label_spans
        )
        batch_entity_spans += _batch_entity_spans
        new_batch_id += _batch_id
        new_batch_text += _batch_text
        new_batch_label_spans += _batch_label_spans
    return batch_entity_spans, new_batch_id, new_batch_text, new_batch_label_spans


def create_entity_spans(
    word_spans: list[WordSpan],
    tokenizer: LukeTokenizer | MLukeTokenizer,
    id: int,
    text: str,
    label_spans: list[LabelSpan],
) -> tuple[list[list[EntitySpan]], list[int], list[str], list[list[LabelSpan]]]:
    """任意の方法で分割したテキストからentity_spansを作成する"""

    max_entity_length = tokenizer.max_entity_length
    max_mention_length = tokenizer.max_mention_length
    subword_lengths = [len(tokenizer.tokenize(word_span["word"])) for word_span in word_spans]

    word_start_positions = [word_span["start"] for word_span in word_spans]
    word_end_positions = [word_span["end"] for word_span in word_spans]

    batch_entity_spans: list[list[EntitySpan]] = []
    entity_spans: list[EntitySpan] = []
    for i_start, start_posision in enumerate(word_start_positions):
        for i_end, end_position in enumerate(word_end_positions[i_start:], start=i_start):
            # エンティティ内のサブワードの数がmax_mention_lengthを超えたらエンティティの始端を進める
            if sum(subword_lengths[i_start : i_end + 1]) > max_mention_length:
                break
            else:
                entity_spans.append((start_posision, end_position))
            # エンティティの数がmax_entity_lengthを超えたらスパンの作成を終了する
            # 溢れた分は新しいデータとして追加する
            if len(entity_spans) >= max_entity_length:
                batch_entity_spans.append(entity_spans)
                entity_spans = []
    if entity_spans:
        batch_entity_spans.append(entity_spans)

    batch_id = [id] * len(batch_entity_spans)
    batch_text = [text] * len(batch_entity_spans)
    batch_label_spans = [label_spans] * len(batch_entity_spans)
    return batch_entity_spans, batch_id, batch_text, batch_label_spans


def batch_tokenize(
    batch_text: list[str],
    batch_entity_spans: list[list[EntitySpan]],
    tokenizer: LukeTokenizer | MLukeTokenizer,
    tokenize_kwargs: dict[str, Any] | None = None,
) -> BatchEncoding:
    tokenize_kwargs = tokenize_kwargs or {}
    return tokenizer(batch_text, entity_spans=batch_entity_spans, **tokenize_kwargs)


def batch_align_labels(
    batch_entity_spans: list[list[EntitySpan]], batch_label_spans: list[list[LabelSpan]], label2id: dict[str, int]
) -> list[list[int]]:
    return [
        align_labels(entity_spans, label_spans, label2id)
        for entity_spans, label_spans in zip(batch_entity_spans, batch_label_spans, strict=True)
    ]


def align_labels(entity_spans: list[EntitySpan], label_spans: list[LabelSpan], label2id: dict[str, int]) -> list[int]:
    labels = [label2id[NON_ENTITY]] * len(entity_spans)
    for label_span in label_spans:
        for i, entity_span in enumerate(entity_spans):
            if label_span["start"] == entity_span[0] and label_span["end"] == entity_span[1]:
                labels[i] = label2id[label_span["label"]]
                break
    return labels


def batch_prepare_labels_for_evaluation(
    batch_example: dict[str, list],
) -> dict[str, list]:
    batch_text, batch_label_spans = batch_example["text"], batch_example["label_spans"]
    batch_iob2_labels = batch_convert_label_spans_to_iob2_labels(batch_text, batch_label_spans)
    return {"iob2_labels": batch_iob2_labels}


def batch_convert_label_spans_to_iob2_labels(
    batch_text: list[str],
    batch_label_spans: list[list[LabelSpan]],
) -> list[list[str]]:
    return [
        convert_label_spans_to_iob2_labels(text, label_spans)
        for text, label_spans in zip(batch_text, batch_label_spans, strict=True)
    ]


def convert_label_spans_to_iob2_labels(
    text: str,
    label_spans: list[LabelSpan],
) -> list[str]:
    """ラベルを一文字単位のIOB2形式のラベルに変換する。(評価の際にテキストの分割方法による差異を吸収するため)"""

    iob2_labels = [NON_ENTITY] * len(text)
    for label_span in label_spans:
        iob2_labels[label_span["start"]] = f"B-{label_span['label']}"
        for i in range(label_span["start"] + 1, label_span["end"]):
            iob2_labels[i] = f"I-{label_span['label']}"
    return iob2_labels
