from typing import TypeAlias, TypedDict

NON_ENTITY = "O"

EntitySpan: TypeAlias = tuple[int, int]


class LabelSpan(TypedDict):
    start: int
    end: int
    entity: str
    label: str


class WordSpan(TypedDict):
    start: int
    end: int
    word: str
