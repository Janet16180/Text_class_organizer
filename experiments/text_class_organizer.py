from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any
from pathlib import Path
import nltk
import os
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import re
from transformers import BertTokenizer, BertModel
import torch
import hdbscan
from sklearn.metrics.pairwise import cosine_distances
import numpy as np
import pickle
from collections import defaultdict
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer

class Error_clusterer_base(ABC):
    def __init__(self, model_path: Path = None, common_dict: Dict[str, str] = None):
        if model_path is None:
            model_path = Path(os.getcwd(), "text_class_organizer_data")
        self.model_path = model_path

        if common_dict is None:
            common_dict = {
                'CLUSTER_SYSTEM_NUMBER': r"\d+",
                'CLUSTER_SYSTEM_HEX_VAL': r"0x[\da-fA-F]+",
                "CLUSTER_SYSTEM_TIME": r"\d+(.\d+)?(s|ms|us|ps|fs)",
            }

        self.common_dict = common_dict

        nltk.download('punkt_tab', download_dir=self.model_path)
        nltk.download('punkt', download_dir=self.model_path)
        nltk.data.path.append(self.model_path)

        # self.tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-cased')
        # self.tokenizer_model = BertModel.from_pretrained('google-bert/bert-base-cased')

        self.tokenizer = SentenceTransformer('all-MiniLM-L6-v2')

    def get_embeddings(self, errors: List[str]) -> torch.Tensor:
        # process_text = list(map(self.clean_data, errors))
        # inputs = self.tokenizer(process_text, return_tensors='pt', padding=True, truncation=True)
        # with torch.no_grad():
        #     outputs = self.tokenizer_model(**inputs)
        # embeddings = outputs.last_hidden_state
        # return embeddings[:, 0, :]

        if not isinstance(errors, list):
            errors = list(errors)

        embeddings = self.tokenizer.encode(errors, convert_to_tensor=True)
        return embeddings


    
    def cleaner_remplace_common(self, text: str) -> str:
        tokenize_text = word_tokenize(text)
        for i, word in enumerate(tokenize_text):
            for remplace, regex in self.common_dict.items():
                if re.search(regex, word):
                    tokenize_text[i] = remplace

        detokenizer = TreebankWordDetokenizer()
        return detokenizer.detokenize(tokenize_text)

    def clean_data(self, text: str, cleaner_methods: List[str] = None) -> str:
        if cleaner_methods is None:
            cleaner_methods = [method for method in dir(self) if callable(getattr(self, method)) and method.startswith("cleaner_")]
        for method in cleaner_methods:
            text = getattr(self, method)(text)
        return text



    def cluster_without_pretraining(self, data: List[str]) -> Tuple[List[str], Dict[int, List[str]]]:
        cls_embeddings = self.get_embeddings(data)
        cosine_distance_matrix = cosine_distances(cls_embeddings)
        clustering = hdbscan.HDBSCAN(min_cluster_size=3, metric='precomputed').fit(cosine_distance_matrix)
        labels = clustering.labels_

        clusters = defaultdict(list)
        for label, log_entry in zip(labels, data):
            clusters[int(label)].append(log_entry)

        noise = clusters.pop(-1, [])
        return noise, clusters

    def save_model(self, path: Path) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self.centroid_indices, f)

    def load_model(self, path: Path) -> None:
        with open(path, 'rb') as f:
            self.centroid_indices = pickle.load(f)

    @abstractmethod
    def fit(self, errors: List[str], labels: List[Any]) -> None:
        pass

    @abstractmethod
    def predict(self, data: List[str]) -> Dict[Any, List[str]]:
        pass


class Error_clusterer_centroid(Error_clusterer_base):
    def __init__(self, model_path: Path = None, common_dict: Dict[str, str] = None):
        super().__init__(model_path, common_dict)
        self.centroid_indices = {}

    def fit(self, errors: List[str], labels: List[Any]) -> None:
        clusters = defaultdict(list)
        for label, error in zip(labels, errors):
            clusters[label].append(error)

        clusters_cls_embeddings = {cluster_id: self.get_embeddings(cluster_data)
                                   for cluster_id, cluster_data in clusters.items()}

        self.centroid_indices = {label: vectors.mean(dim=0) 
                                 for label, vectors in clusters_cls_embeddings.items()}

    def predict(self, data: List[str]) -> Dict[Any, List[str]]:
        cls_embeddings = self.get_embeddings(data)
        result = []

        for single_cls_embedding, log_entry in zip(cls_embeddings, data):
            cosine_distances = {label: torch.nn.functional.cosine_similarity(single_cls_embedding, centroid, dim=0)
                                for label, centroid in self.centroid_indices.items()}
            closest_label = max(cosine_distances, key=cosine_distances.get)
            result.append(closest_label)

        return result


class Error_clusterer_naive_bayes(Error_clusterer_base):
    def __init__(self, model_path: Path = None, common_dict: Dict[str, str] = None):
        super().__init__(model_path, common_dict)


    def fit(self, errors: List[str], labels: List[Any]) -> None:
        cls_embbedings = self.get_embeddings(errors)
        self.scaler = MinMaxScaler()
        cls_embbedings = self.scaler.fit_transform(cls_embbedings)

        self.classifier = MultinomialNB()
        self.classifier.fit(cls_embbedings, labels)

    def predict(self, data):
        cls_embbedings = self.get_embeddings(data)

        cls_embbedings = self.scaler.fit_transform(cls_embbedings)
        return self.classifier.predict(cls_embbedings)
    
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GridSearchCV

class Error_clusterer_xgboost(Error_clusterer_base):
    def __init__(self, model_path: Path = None, common_dict: Dict[str, str] = None):
        super().__init__(model_path, common_dict)

    def fit(self, errors: List[str], labels: List[Any]) -> None:
        cls_embbedings = self.get_embeddings(errors)
        self.classifier = XGBClassifier()

        param_grid = {
            'max_depth': [3, 5, 7],
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2]
        }
        self.labels_to_int = {label: i for i, label in enumerate(set(labels))}
        self.int_to_labels = {i: label for i, label in enumerate(set(labels))}


        xgb_clf = XGBClassifier()
        grid_search = GridSearchCV(xgb_clf, param_grid, cv=3, scoring='accuracy')
        grid_search.fit(cls_embbedings, [self.labels_to_int[label] for label in labels])  # Assuming 'labels' is your target variable

        best_model = grid_search.best_estimator_
        self.classifier = best_model

    def predict(self, data):
        cls_embbedings = self.get_embeddings(data)
        predict = self.classifier.predict(cls_embbedings)
        predict = [self.int_to_labels[label] for label in predict]
        return predict


