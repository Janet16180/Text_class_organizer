Error Clusterer Quickstart Guide
================================

The `Error Clusterer` library provides tools for clustering error messages using various machine learning algorithms, with sentence embeddings as a core component. This guide will walk you through a quick start on how to use these implementations effectively.

Getting Started
---------------

First, ensure you've installed the necessary prerequisites, including the relevant Python packages like `sentence_transformers`, `scikit-learn`, and `torch`.

Error Clusterer Implementations
-------------------------------

This library offers different clustering methodologies built upon the `Error_clusterer_base` class, which leverages sentence embeddings generated by pre-trained models.

1. **Error_clusterer_centroid**

   This implementation is extremely fast, making it ideal for scenarios where quick predictions are crucial.

   .. code-block:: python

       from text_class_organizer import Error_clusterer_centroid

       # Initialize the Centroid Error Clusterer
       clusterer = Error_clusterer_centroid()

       # Extensive example error messages
       errors = [
           "Error: Failed to connect to database.",
           "Warning: Low disk space.",
           "Error: Out of memory.",
           "Alert: Unauthorized access attempt detected.",
           "Info: Database backup completed successfully.",
           "Error: Invalid input format.",
           "Critical: System overheating warning.",
           "Warning: Password length is weak.",
           "Error: Network timeout while accessing server.",
           "Info: User logged in from new device.",
       ]

       # Corresponding labels
       labels = ["database", "system", "memory", "security", 
                 "maintenance", "input", "system", "security",
                 "network", "security"]

       # Fit the model on the error messages
       clusterer.fit(errors, labels)

       # Predict labels for new errors
       new_errors = [
           "Critical: No database connection.",
           "Disk space is critically low.",
           "Unauthorized login attempt blocked.",
           "Invalid data format submitted.",
           "System memory exceeded.",
       ]

       predicted_labels = clusterer.predict(new_errors)
       print(predicted_labels)  # Output might be ['database', 'system', 'security', 'input', 'memory']

2. **Error_clusterer_naive_bayes**

   Generally, this approach yields good results with its simple, efficient Naive Bayes algorithm.

   .. code-block:: python

       from text_class_organizer import Error_clusterer_naive_bayes

       # Initialize the Naive Bayes Error Clusterer
       clusterer = Error_clusterer_naive_bayes()

       # Use the same extensive example as in the centroid example

3. **Additional Implementations**

The library also provides `Error_clusterer_svm`, `Error_clusterer_random_forest`, and `Error_clusterer_xgboost` for more advanced clustering needs. These implementations preprocess data using techniques like SVD and Scaling for improved performance on complex datasets.

The `Error Clusterer` library is flexible and allows for selecting different clustering techniques based on your specific requirements. For quick processing, `Error_clusterer_centroid` is recommended, while `Error_clusterer_naive_bayes` provides robust results for general use cases.
