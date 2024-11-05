Error Clusterer Base
=====================

The `Error_clusterer_base` is an abstract base class designed to facilitate the clustering of error messages using sentence embeddings. It leverages the capabilities of pre-trained sentence transformer models to generate these embeddings.

.. currentmodule:: text_class_organizer.error_clusterer.error_clusterer_base

Class Overview
--------------

.. autoclass:: Error_clusterer_base
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :inherited-members:
   :noindex:


Below is an example of how to implement and use the `Error_clusterer_base` class:

.. code-block:: python

    from text_class_organizer.error_clusterer.error_clusterer_base import Error_clusterer_base
    from sentence_transformers import SentenceTransformer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.preprocessing import MinMaxScaler
    import torch
    from typing import List, Any

    class CustomErrorClusterer(Error_clusterer_base):
        def __init__(self, sentence_model: str = "all-MiniLM-L6-v2", **kwargs) -> None:
            super().__init__(sentence_model)
            self.classifier = MultinomialNB(**kwargs)
            self.scaler = None

        def preprocess_features(self, cls_embeddings: torch.Tensor) -> torch.Tensor:
            if self.scaler is not None:
                cls_embeddings = self.scaler.transform(cls_embeddings)
            else:
                self.scaler = MinMaxScaler()
                cls_embeddings = self.scaler.fit_transform(cls_embeddings)
            return cls_embeddings

        def fit(self, errors: List[str], labels: List[Any]) -> None:
            cls_embeddings = self.get_embeddings(errors)
            cls_embeddings = self.preprocess_features(cls_embeddings)
            self.classifier.fit(cls_embeddings, labels)

        def predict(self, data: List[str]) -> List[Any]:
            cls_embeddings = self.get_embeddings(data)
            cls_embeddings = self.preprocess_features(cls_embeddings)
            return self.classifier.predict(cls_embeddings)

    # Example usage of the CustomErrorClusterer
    clusterer = CustomErrorClusterer()

    # Example error messages
    errors = [
        "Error: Failed to connect to database.",
        "Warning: Low disk space.",
        "Error: Out of memory.",
    ]
    
    # Corresponding labels
    labels = ["database", "system", "memory"]

    # Fit the model on the error messages
    clusterer.fit(errors, labels)

    # Predict labels for new errors
    new_errors = [
        "Critical: No database connection.",
        "Disk space is critically low.",
    ]

    predicted_labels = clusterer.predict(new_errors)
    print(predicted_labels)  # Output might be ['database', 'system']

The `Error_clusterer_naive_bayes` class is an implementation of `Error_clusterer_base` that uses a Naive Bayes classifier. While this provides an immediate, simple clustering solution, users are encouraged to implement custom models by extending `Error_clusterer_base`. This flexibility allows for the use of different algorithms and techniques as needed.


Error Clusterer Implementations
===============================

This section covers various implementations of the `Error_clusterer_base`, providing distinct methodologies for clustering error messages using different machine learning algorithms.

.. currentmodule:: text_class_organizer.error_clusterer.error_clusterer_implementations

Class Implementations
---------------------

Error_clusterer_centroid
------------------------

`Error_clusterer_centroid` uses a centroid-based clustering approach. Each cluster is represented by the mean of the embeddings from errors associated with a particular label.

.. autoclass:: Error_clusterer_centroid
   :members:
   :show-inheritance:

Error_clusterer_naive_bayes
---------------------------

`Error_clusterer_naive_bayes` is an implementation of `Error_clusterer_base` that applies a Naive Bayes classifier to cluster error messages based on their embeddings.

.. autoclass:: Error_clusterer_naive_bayes
   :members:
   :show-inheritance:

Error_clusterer_svm
-------------------

`Error_clusterer_svm` uses a Support Vector Machine (SVM) for classification. It preprocesses embeddings using StandardScaler and TruncatedSVD to handle high-dimensional data efficiently.

.. autoclass:: Error_clusterer_svm
   :members:
   :show-inheritance:

Error_clusterer_random_forest
-----------------------------

`Error_clusterer_random_forest` leverages the Random Forest algorithm for clustering. It optionally applies TruncatedSVD for dimensionality reduction.

.. autoclass:: Error_clusterer_random_forest
   :members:
   :show-inheritance:

Error_clusterer_xgboost
-----------------------

`Error_clusterer_xgboost` implements an XGBoost classifier for clustering. It preprocesses features using both StandardScaler and TruncatedSVD, and maps labels to integer indices for compatibility with the XGBoost library.

.. autoclass:: Error_clusterer_xgboost
   :members:
   :show-inheritance:

Usage Example
-------------

.. code-block:: python

    from text_class_organizer import Error_clusterer_naive_bayes

    # Initialize the Naive Bayes Error Clusterer
    clusterer = Error_clusterer_naive_bayes()

    # Example error messages
    errors = [
        "Error: Failed to connect to database.",
        "Warning: Low disk space.",
        "Error: Out of memory.",
    ]

    # Corresponding labels
    labels = ["database", "system", "memory"]

    # Fit the model on the error messages
    clusterer.fit(errors, labels)

    # Predict labels for new errors
    new_errors = [
        "Critical: No database connection.",
        "Disk space is critically low.",
    ]

    predicted_labels = clusterer.predict(new_errors)
    print(predicted_labels)  # Output might be ['database', 'system']