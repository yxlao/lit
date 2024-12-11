import pickle
from abc import ABC
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Iterable

import numpy as np


class BaseContainer(ABC):
    """
    Abstract base class that implements the save() and load() methods by calling
    to_dict() and from_dict(). Child classes must implement to_dict() and
    from_dict(). All np.float32 checks are enforced before saving.
    """

    def __post_init__(self):
        BaseContainer._assert_float32(self)

    def to_dict(self):
        raise NotImplementedError(f"{cls.__name__} must implement from_dict method")

    @classmethod
    def from_dict(cls, dict_data: dict):
        raise NotImplementedError(f"{cls.__name__} must implement from_dict method")

    def save(self, path: Path, verbose=False):
        data = self.to_dict()
        BaseContainer._assert_float32(data)
        with open(path, "wb") as file:
            pickle.dump(data, file)
        if verbose:
            print(f"Saved {self.__class__.__name__} to {path}")

    @classmethod
    def load(cls, path: Path):
        with open(path, "rb") as file:
            data = pickle.load(file)
        return cls.from_dict(data)

    @staticmethod
    def _assert_float32(value: Any, name: str = None):
        if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.floating):
            if value.dtype != np.float32:
                if name is None:
                    raise ValueError(f"Array must be np.float32, got {value.dtype}.")
                else:
                    raise ValueError(f"{name} must be np.float32, got {value.dtype}.")
        elif isinstance(value, BaseContainer):
            BaseContainer._assert_float32(value.to_dict())
        elif isinstance(value, dict):
            for k, v in value.items():
                BaseContainer._assert_float32(k)
                BaseContainer._assert_float32(v, name=str(k))
        elif isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            for item in value:
                BaseContainer._assert_float32(item)
