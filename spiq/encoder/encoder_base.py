import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Iterable, Iterator, Any, List, Callable
from sklearn.base import BaseEstimator
from tqdm import tqdm
from spiq.streamer.data_streamer import DataStreamer

class PQEncoderBase(BaseEstimator, ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'PQEncoderBase':
        """Fit the encoder to the training data."""
        pass

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Encode input vectors into PQ codes."""
        pass

    @abstractmethod
    def inverse_transform(self, X: np.ndarray, binary: bool = False) -> np.ndarray:
        """Reconstruct the original vectors from PQ codes."""
        pass

    DataStreamer._process_chunk()