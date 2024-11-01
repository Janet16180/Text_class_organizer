from .error_clusterer_base import Error_clusterer_base
from typing import Dict, Any, List
from collections import defaultdict
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

class Error_clusterer_centroid(Error_clusterer_base):
    def __init__(self, sentence_model: str = "all-MiniLM-L6-v2"):
        """
        Initializes the Error_clusterer_centroid class with a specified sentence model.

        Parameters
        ----------
        sentence_model : str, optional
            The sentence model to be used, by default "all-MiniLM-L6-v2"
        """
        super().__init__(sentence_model)
        self.centroid_indices = {}

    def fit(self, errors: List[str], labels: List[Any]) -> None:
        """
        Fits the centroid-based clustering model using the given errors and labels.

        Parameters
        ----------
        errors : List[str]
            A list of error strings to be clustered.
        labels : List[Any]
            A list of labels corresponding to each error.
        """
        clusters = defaultdict(list)
        for label, error in zip(labels, errors):
            clusters[label].append(error)

        clusters_cls_embeddings = {cluster_id: self.get_embeddings(cluster_data)
                                   for cluster_id, cluster_data in clusters.items()}

        self.centroid_indices = {label: vectors.mean(dim=0) 
                                 for label, vectors in clusters_cls_embeddings.items()}

    def predict(self, data: List[str]) -> List[Any]:
        """
        Predicts the closest cluster label for each item in the given data.

        Parameters
        ----------
        data : List[str]
            A list of data strings for which to predict the closest cluster labels.

        Returns
        -------
        List[Any]
            A list of predicted labels for the input data.
        """
        cls_embeddings = self.get_embeddings(data)
        result = []

        for single_cls_embedding in cls_embeddings:
            cosine_distances = {label: torch.nn.functional.cosine_similarity(single_cls_embedding, centroid, dim=0)
                                for label, centroid in self.centroid_indices.items()}
            closest_label = max(cosine_distances, key=cosine_distances.get)
            result.append(closest_label)

        return result

    def preprocess_features(self):
        """
        Placeholder for feature preprocessing.
        Returns
        -------
        None
        """
        return None


class Error_clusterer_naive_bayes(Error_clusterer_base):
    def __init__(self, sentence_model: str = "all-MiniLM-L6-v2", **kwargs) -> None:
        """
        Initializes the Error_clusterer_naive_bayes class.

        Parameters
        ----------
        sentence_model : str, optional
            The sentence model to be used, by default "all-MiniLM-L6-v2"
        """
        super().__init__(sentence_model)
        self.classifier = MultinomialNB(**kwargs)
        self.scaler = None
        self.classifier = None

    def preprocess_features(self, cls_embbedings: torch.Tensor) -> torch.Tensor:
        """
        Preprocesses class embeddings using MinMaxScaler.

        If a scaler is already present, it transforms the embeddings;
        otherwise, it initializes a MinMaxScaler and fits the embeddings.

        Parameters
        ----------
        cls_embbedings : torch.Tensor
            The class embeddings to be preprocessed.

        Returns
        -------
        torch.Tensor
            The preprocessed class embeddings.
        """
        if self.scaler is not None:
            cls_embbedings = self.scaler.transform(cls_embbedings)
        else:
            self.scaler = MinMaxScaler()
            cls_embbedings = self.scaler.fit_transform(cls_embbedings)

        return cls_embbedings

    def fit(self, errors: List[str], labels: List[Any]) -> None:
        """
        Fits the Naive Bayes classifier using the given errors and labels.

        Parameters
        ----------
        errors : List[str]
            A list of error strings to be clustered.
        labels : List[Any]
            A list of labels corresponding to each error.
        **kwargs
            Additional arguments for the MultinomialNB classifier.
        """
        cls_embbedings = self.get_embeddings(errors)
        cls_embbedings = self.preprocess_features(cls_embbedings)
        self.classifier.fit(cls_embbedings, labels)

    def predict(self, data: List[str]) -> List[Any]:
        """
        Predicts the labels for the given data using the trained classifier.

        Parameters
        ----------
        data : List[str]
            A list of data strings for which to predict labels.

        Returns
        -------
        List[Any]
            A list of predicted labels for the input data.
        """
        cls_embbedings = self.get_embeddings(data)
        cls_embbedings = self.preprocess_features(cls_embbedings)
        return self.classifier.predict(cls_embbedings)


class Error_clusterer_svm(Error_clusterer_base):
    def __init__(self, sentence_model: str = "all-MiniLM-L6-v2", **kwargs) -> None:
        """
        Initializes the Error_clusterer_svm class with a specified sentence model.

        Parameters
        ----------
        sentence_model : str, optional
            The sentence model to be used, by default "all-MiniLM-L6-v2".
        """
        super().__init__(sentence_model)
        self.classifier = SVC(**kwargs)
        self.scaler = None
        self.svd = None

    def preprocess_features(self, cls_embbedings: torch.Tensor, n_components: int = 100) -> torch.Tensor:
        """
        Preprocesses class embeddings using StandardScaler and TruncatedSVD.

        If the scaler and SVD are already initialized, it transforms the embeddings;
        otherwise, it initializes them and fits the embeddings.

        Parameters
        ----------
        cls_embbedings : torch.Tensor
            The class embeddings to be preprocessed.
        n_components : int, optional
            Number of components for SVD, by default 100.

        Returns
        -------
        torch.Tensor
            The preprocessed class embeddings.
        """
        if self.scaler is not None and self.svd is not None:
            cls_embbedings = self.scaler.transform(cls_embbedings)
            cls_embbedings = self.svd.transform(cls_embbedings)
        else:
            self.scaler = StandardScaler()
            self.svd = TruncatedSVD(n_components=n_components)
            cls_embbedings = self.scaler.fit_transform(cls_embbedings)
            cls_embbedings = self.svd.fit_transform(cls_embbedings)

        return cls_embbedings
    
    def fit(self, errors: List[str], labels: List[Any]) -> None:
        """
        Fits the SVM classifier with the given errors and labels.

        Parameters
        ----------
        errors : List[str]
            A list of error strings to be clustered.
        labels : List[Any]
            A list of labels corresponding to each error.
        """
        cls_embeddings = self.get_embeddings(errors)
        cls_embeddings = self.preprocess_features(cls_embeddings)
        self.classifier.fit(cls_embeddings, labels)

    def predict(self, data: List[str]) -> List[Any]:
        """
        Predicts the labels for the given data using the trained SVM classifier.

        Parameters
        ----------
        data : List[str]
            A list of data strings for which to predict labels.

        Returns
        -------
        List[Any]
            A list of predicted labels for the input data.
        """
        cls_embeddings = self.get_embeddings(data)
        cls_embeddings = self.preprocess_features(cls_embeddings)
        return self.classifier.predict(cls_embeddings)

    def predict_proba(self, data: List[str]) -> torch.Tensor:
        """
        Predicts class probabilities for the given data using the trained SVM classifier.

        Parameters
        ----------
        data : List[str]
            A list of data strings for which to predict class probabilities.

        Returns
        -------
        torch.Tensor
            An array of predicted class probabilities for each input data.
        """
        cls_embeddings = self.get_embeddings(data)
        cls_embeddings = self.scaler.transform(cls_embeddings)
        return self.classifier.predict_proba(cls_embeddings)
    

class Error_clusterer_random_forest(Error_clusterer_base):
    def __init__(self, sentence_model: str = "all-MiniLM-L6-v2", **kwargs) -> None:
        """
        Initializes the Error_clusterer_random_forest class with a specified sentence model.

        Parameters
        ----------
        sentence_model : str, optional
            The sentence model to be used, by default "all-MiniLM-L6-v2".
        """
        super().__init__(sentence_model)
        self.classifier = RandomForestClassifier(**kwargs)
        self.svd = None

    def preprocess_features(self, cls_embeddings: torch.Tensor, n_components: int = 100) -> torch.Tensor:
        """
        Preprocesses class embeddings using TruncatedSVD.

        If the SVD transformer is already initialized, it transforms the embeddings;
        otherwise, it initializes TruncatedSVD and fits the embeddings.

        Parameters
        ----------
        cls_embeddings : torch.Tensor
            The class embeddings to be preprocessed.
        n_components : int, optional
            Number of components for SVD, by default 100.

        Returns
        -------
        torch.Tensor
            The preprocessed class embeddings.
        """
        if self.svd is not None:
            cls_embeddings = self.svd.transform(cls_embeddings)
        else:
            self.svd = TruncatedSVD(n_components=n_components)
            cls_embeddings = self.svd.fit_transform(cls_embeddings)

        return cls_embeddings

    def fit(self, errors: List[str], labels: List[Any]) -> None:
        """
        Fits the Random Forest classifier using the provided errors and labels.

        Parameters
        ----------
        errors : List[str]
            A list of error strings to be clustered.
        labels : List[Any]
            A list of labels corresponding to each error.
        **kwargs
            Additional keyword arguments for the RandomForestClassifier.
        """
        cls_embeddings = self.get_embeddings(errors)
        cls_embeddings = self.preprocess_features(cls_embeddings)

        self.classifier.fit(cls_embeddings, labels)

    def predict(self, data: List[str]) -> List[Any]:
        """
        Predicts the labels for the given data using the trained Random Forest classifier.

        Parameters
        ----------
        data : List[str]
            A list of data strings for which to predict labels.

        Returns
        -------
        List[Any]
            A list of predicted labels for the input data.
        """
        cls_embeddings = self.get_embeddings(data)
        cls_embeddings = self.preprocess_features(cls_embeddings)

        return self.classifier.predict(cls_embeddings)
    

class Error_clusterer_xgboost(Error_clusterer_base):
    def __init__(self, sentence_model: str = "all-MiniLM-L6-v2", **kwargs) -> None:
        """
        Initializes the Error_clusterer_xgboost class with a specified sentence model.

        Parameters
        ----------
        sentence_model : str, optional
            The sentence model to be used, by default "all-MiniLM-L6-v2".
        **kwargs
            Additional keyword arguments for the XGBClassifier.
        """
        super().__init__(sentence_model)
        self.classifier = XGBClassifier(**kwargs)
        self.scaler = None
        self.svd = None

    def preprocess_features(self, cls_embbedings: torch.Tensor, n_components: int = 100) -> torch.Tensor:
        """
        Preprocesses class embeddings using StandardScaler and TruncatedSVD.

        If the scaler and SVD are already initialized, it transforms the embeddings;
        otherwise, it initializes them and fits the embeddings.

        Parameters
        ----------
        cls_embbedings : torch.Tensor
            The class embeddings to be preprocessed.
        n_components : int, optional
            Number of components for SVD, by default 100.

        Returns
        -------
        torch.Tensor
            The preprocessed class embeddings.
        """
        if self.scaler is not None and self.svd is not None:
            cls_embbedings = self.scaler.transform(cls_embbedings)
            cls_embbedings = self.svd.transform(cls_embbedings)
        else:
            self.scaler = StandardScaler()
            self.svd = TruncatedSVD(n_components=n_components)
            cls_embbedings = self.scaler.fit_transform(cls_embbedings)
            cls_embbedings = self.svd.fit_transform(cls_embbedings)
        return cls_embbedings

    def fit(self, errors: List[str], labels: List[Any]) -> None:
        """
        Fits the XGBoost classifier with the given errors and labels.

        Parameters
        ----------
        errors : List[str]
            A list of error strings to be clustered.
        labels : List[Any]
            A list of labels corresponding to each error.
        """
        cls_embbedings = self.get_embeddings(errors)
        cls_embbedings = self.preprocess_features(cls_embbedings)

        self.int_to_labels = {i: label for i, label in enumerate(set(labels))}
        self.labels_to_int = {label: i for i, label in self.int_to_labels.items()}

        labels = [self.labels_to_int[label] for label in labels]
        self.classifier.fit(cls_embbedings, labels)

    def predict(self, data: List[str]) -> List[Any]:
        """
        Predicts the labels for the given data using the trained XGBoost classifier.

        Parameters
        ----------
        data : List[str]
            A list of data strings for which to predict labels.

        Returns
        -------
        List[Any]
            A list of predicted labels for the input data.
        """
        cls_embbedings = self.get_embeddings(data)
        cls_embbedings = self.preprocess_features(cls_embbedings)

        predict = self.classifier.predict(cls_embbedings)
        predict = [self.int_to_labels[label] for label in predict]
        return predict
