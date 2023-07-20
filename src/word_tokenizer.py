from abc import ABC, abstractmethod
from typing import Any

from src.constant import WordSpan


class WordTokenizer(ABC):
    tokenizer: Any

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, text: str) -> list[WordSpan]:
        pass


class SudachiTokenizer(WordTokenizer):
    def __init__(self, dict: str = "full", mode: str = "C"):
        import sudachipy

        mode_dict = {
            "A": sudachipy.Tokenizer.SplitMode.A,
            "B": sudachipy.Tokenizer.SplitMode.B,
            "C": sudachipy.Tokenizer.SplitMode.C,
        }
        self.tokenizer = sudachipy.Dictionary(dict=dict).create(mode=mode_dict[mode])

    def __call__(self, text: str) -> list[WordSpan]:
        return [
            {"start": word.begin(), "end": word.end(), "word": word.surface()}
            for word in self.tokenizer.tokenize(text)
        ]


class AutoWordTokenizer:
    @classmethod
    def from_config(cls, type: str | None = None, **kwargs) -> WordTokenizer | None:
        if type is None:
            return None
        elif type == "sudachi":
            return SudachiTokenizer(**kwargs)
        else:
            raise ValueError(f"Invalid tokenizer type: {type}")
