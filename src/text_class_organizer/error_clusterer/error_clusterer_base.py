from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any
from pathlib import Path
import pickle
import os
import torch
from sentence_transformers import SentenceTransformer

class Error_clusterer_base(ABC):
    """
    A base class for clustering error messages using sentence embeddings.
    
    Attributes
    ----------
    tmp_dir : Path
        Directory for storing temporary data.
    tokenizer : SentenceTransformer
        Pre-trained sentence transformer model for generating embeddings.
    
    Parameters
    ----------
    sentence_model : str, optional
        Name of the pre-trained sentence transformer model to use, by default "all-MiniLM-L6-v2".
    """
    def __init__(self, sentence_model: str = "all-MiniLM-L6-v2") -> None:

        self.tokenizer = SentenceTransformer(sentence_model)

    def get_embeddings(self, errors: List[str]) -> torch.Tensor:
        """
        Generate sentence embeddings for a list of error messages.

        Parameters
        ----------
        errors : list of str
            List of error strings for which embeddings are to be generated.

        Returns
        -------
        torch.Tensor
            Tensor of sentence embeddings.
        """

        if not isinstance(errors, list):
            errors = list(errors)

        embeddings = self.tokenizer.encode(errors, convert_to_tensor=True)
        return embeddings

    def clean_data(self, text: str) -> str:
        """
        Clean text data by stripping whitespace and converting to lowercase.

        Parameters
        ----------
        text : str
            The text to be cleaned.

        Returns
        -------
        str
            Cleaned text.
        """
        return text.strip().lower()

    def save_model(self, path: Path) -> None:
        """
        Save the model's centroid indices to a file.

        Parameters
        ----------
        path : Path
            Path to the file where the model will be saved.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.centroid_indices, f)

    def load_model(self, path: Path) -> None:
        """
        Load model centroid indices from a file.

        Parameters
        ----------
        path : Path
            Path to the file from which the model will be loaded.
        """
        with open(path, 'rb') as f:
            self.centroid_indices = pickle.load(f)

    @abstractmethod
    def fit(self, errors: List[str], labels: List[Any]) -> None:
        """
        Abstract method to fit the clustering model.

        Parameters
        ----------
        errors : list of str
            List of error messages to fit the model on.
        labels : list of Any
            List of labels corresponding to the error messages.
        """
        pass

    @abstractmethod
    def predict(self, data: List[str]) -> List[Any]:
        """
        Abstract method to predict the cluster labels for given data.

        Parameters
        ----------
        data : list of str
            List of data strings for which predictions are to be made.

        Returns
        -------
        list of Any
            List of predicted labels for the input data.
        """
        pass