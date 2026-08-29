"""Plugin registry — simple dict-based, per spec Section 5.12."""
from __future__ import annotations

from typing import Iterable, Protocol, runtime_checkable

from backend.app.schemas import JudgeResult, Record


@runtime_checkable
class FeatureExtractorPlugin(Protocol):
    name: str

    def extract(self, answer: str) -> dict[str, float]: ...


@runtime_checkable
class JudgePlugin(Protocol):
    name: str

    def judge(
        self,
        answer: str,
        fingerprint_summary: dict,
        ambiguous_patterns: list[dict],
    ) -> JudgeResult: ...


@runtime_checkable
class DatasetLoaderPlugin(Protocol):
    name: str

    def load(self) -> Iterable[Record]: ...


_feature_extractors: dict[str, FeatureExtractorPlugin] = {}
_judges: dict[str, JudgePlugin] = {}
_dataset_loaders: dict[str, DatasetLoaderPlugin] = {}


def register_feature_extractor(plugin: FeatureExtractorPlugin) -> None:
    _feature_extractors[plugin.name] = plugin


def register_judge(plugin: JudgePlugin) -> None:
    _judges[plugin.name] = plugin


def register_dataset_loader(plugin: DatasetLoaderPlugin) -> None:
    _dataset_loaders[plugin.name] = plugin


def get_feature_extractor(name: str) -> FeatureExtractorPlugin:
    if name not in _feature_extractors:
        raise KeyError(f"No feature extractor registered with name '{name}'")
    return _feature_extractors[name]


def get_judge(name: str) -> JudgePlugin:
    if name not in _judges:
        raise KeyError(f"No judge registered with name '{name}'")
    return _judges[name]


def get_dataset_loader(name: str) -> DatasetLoaderPlugin:
    if name not in _dataset_loaders:
        raise KeyError(f"No dataset loader registered with name '{name}'")
    return _dataset_loaders[name]


def list_dataset_loaders() -> list[str]:
    return list(_dataset_loaders.keys())
