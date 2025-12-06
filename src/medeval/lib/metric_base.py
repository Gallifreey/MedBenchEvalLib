from abc import ABC, abstractmethod
import numpy as np

from medeval.lib.utils import is_installed


class Metric(ABC):

    def __init__(self):
        self._default = {}
        self._default_factory = {}

    def add_state(self, name: str, default=None, default_factory=None) -> None:
        if not hasattr(self, '_default'):
            raise AttributeError('Please call super().__init__() first.')
        if default is None:
            self._default_factory[name] = default_factory
            assert name not in self._default, f'self._default: {self._default}'
            default = default_factory()
        else:
            self._default[name] = default
            assert name not in self._default_factory, f'self._default_factory: {self._default_factory}'
        setattr(self, name, default)

    def reset(self):
        for k, v in self._default.items():
            setattr(self, k, v)
        for k, v in self._default_factory.items():
            setattr(self, k, v())

    @abstractmethod
    def update(self, *args, **kwargs):
        pass

    @abstractmethod
    def compute(self):
        pass


class MeanMetric(Metric):

    def __init__(self, nan_value=0):
        super().__init__()
        self.nan_value = nan_value
        self.add_state('state', default=0.)
        self.add_state('count', default=0)
        self.state = 0
        self.count = 0

    def update(self, state):
        if is_installed("torch"):
            import torch
            if isinstance(state, torch.Tensor):
                state = state.tolist()

        if isinstance(state, np.ndarray):
            state = state.tolist()

        if isinstance(state, (list, tuple)):
            count = len(state)
            state = sum(state)
        else:
            count = 1

        self.state += state
        self.count += count

    def compute(self):
        return {
            'value': self.state / self.count if self.count > 0 else self.nan_value,
        }

    def __str__(self):
        return self.compute()["value"]
