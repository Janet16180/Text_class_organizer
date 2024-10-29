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


class Error_clusterer_base(ABC):
    def __init__(self, model_path: Path = None, common_dict: Dict[str, str] = None):
        if model_path is None:
            model_path = Path(os.getcwd(), "text_class_organizer_data")

        self.model_path = model_path

        if common_dict is None:
            common_dict = {
                'CLUSTER_SYSTEM_NUMBER': r"\d+",
                'CLUSTER_SYSTEM_HEX_VAL': r"0x[\da-fA-F]+",
                "CLUSTER_SYSTEM_TIME": r"\d+(.\d+){0,1}(s|ms|us|ps|fs)",
            }

        self.common_dict = common_dict

        nltk.download('punkt_tab', download_dir=self.model_path)
        nltk.download('punkt', download_dir=self.model_path)

        nltk.data.path.append(self.model_path)

        self.tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-cased')
        self.model = BertModel.from_pretrained('google-bert/bert-base-cased')

        self.centroid_indices = {}

        
    def cleaner_remplace_common(self, text: str) -> str:
        tokenize_text = word_tokenize(text)

        for i, word in enumerate(tokenize_text):
            for remplace, regex in self.common_dict.items():
                match = re.search(regex, word)
                if match:
                    tokenize_text[i] = remplace

        detokenizer = TreebankWordDetokenizer()
        detokenizer_text = detokenizer.detokenize(tokenize_text)
        return detokenizer_text
    
    def clean_data(self, text: str, cleaner_methods: List[str] = None) -> str:

        if cleaner_methods is None:
            cleaner_methods = [method for method in dir(self) if callable(getattr(self, method)) and method.startswith("cleaner_")]

        for method in cleaner_methods:
            text = getattr(self, method)(text)

        return text


    def cluster_without_pretraining(self, data: List[str]) -> Tuple[List[str], Dict[int, List[str]]]:
        # tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-cased', torch_dtype=torch.float16)

        process_text = list(map(self.clean_data, data))
        inputs = self.tokenizer(process_text, return_tensors='pt', padding=True, truncation=True)

        with torch.no_grad():
            outputs = self.model(**inputs, )

        embeddings = outputs.last_hidden_state
        cls_embeddings = embeddings[:, 0, :]

        cosine_distance_matrix = cosine_distances(cls_embeddings)
        cosine_distance_matrix = cosine_distance_matrix.astype(np.float64)
        clustering = hdbscan.HDBSCAN(min_cluster_size=3, metric='precomputed').fit(cosine_distance_matrix)

        labels = clustering.labels_

        clusters = {}
        for label, log_entry in zip(labels, data):
            label = int(label)
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(log_entry)


        noice = []
        if -1 in clusters:
            noice = clusters[-1]
            del clusters[-1]


        return (noice, clusters)
    
    def train_model(self, clusters: Dict[Any, List[str]]) -> None:

        clusters_cls_embeddings = {}
        for cluster_id, cluster_data in clusters.items():
            process_text = list(map(self.clean_data, cluster_data))
            inputs = self.tokenizer(process_text, return_tensors='pt', padding=True, truncation=True)

            with torch.no_grad():
                outputs = self.model(**inputs)

            embeddings = outputs.last_hidden_state
            cls_embeddings = embeddings[:, 0, :]

            clusters_cls_embeddings[cluster_id] = cls_embeddings

        print(cls_embeddings.shape)

        
        centroid_indices = {
            label: vectors.mean(dim=0) for label, vectors in clusters_cls_embeddings.items()
        }
        self.centroid_indices = centroid_indices

    
    def cluster_errors(self, data: List[str]) -> Dict[Any, List[str]]:
        process_text = list(map(self.clean_data, data))
        inputs = self.tokenizer(process_text, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)

        embeddiings = outputs.last_hidden_state
        cls_embeddings = embeddiings[:, 0, :]

        clusters_result = {label: [] for label in self.centroid_indices.keys()}

        for single_cls_embedding, log_entry in zip(cls_embeddings, data):
            cosine_distances = {}
            for label, centroid in self.centroid_indices.items():
                cosine_distances[label] = torch.nn.functional.cosine_similarity(single_cls_embedding, centroid, dim=0)
            closest_label = max(cosine_distances, key=cosine_distances.get)
            clusters_result[closest_label].append(log_entry)

        return clusters_result


    def save_model(self, path: Path) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self.centroid_indices, f)

    def load_model(self, path: Path) -> None:
        with open(path, 'rb') as f:
            self.centroid_indices = pickle.load(f)

        


    


class Error_clusterer(Error_clusterer_base):
    def __init__(self, model_path: Path = None, common_dict: Dict[str, str] = None):
        super().__init__(model_path, common_dict)