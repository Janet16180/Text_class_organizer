# Text Class Organizer

Text Class Organizer is a Python package that helps organize text files into different categories based on their content. Using advanced machine learning algorithms and sentence embeddings, it offers various clustering methodologies for effectively grouping error messages and other textual data.

## Project Status

This project is in its very early stages and is currently in beta. It is not yet available on PyPI. Feedback and contributions are welcome to improve and refine its capabilities.

## Documentation

Detailed documentation is available at: https://janet16180.github.io/Text_class_organizer/build/html/index.html

## Installation

Since the package is not yet available on PyPI, you need to clone the repository and install it manually. Here's how you can set it up:

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd text_class_organizer
   ```

2. Create a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows, use `env\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   python -m pip install -r requirements.txt
   ```

4. Install the package:
   ```bash
   python -m pip install .
   ```

## Quick Usage Example

Below is a quick example showing how to use the `Error_clusterer_centroid` for clustering error messages:

```python
from text_class_organizer import Error_clusterer_centroid

# Initialize the Centroid Error Clusterer
clusterer = Error_clusterer_centroid()

# Example error messages
errors = [
    "Error: Failed to connect to database.",
    "Warning: Low disk space.",
    "Error: Out of memory.",
    "Alert: Unauthorized access attempt detected.",
    "Info: Database backup completed successfully.",
]

# Corresponding labels
labels = ["database", "system", "memory", "security", "maintenance"]

# Fit the model on the error messages
clusterer.fit(errors, labels)

# Predict labels for new error messages
new_errors = [
    "Critical: No database connection.",
    "Disk space is critically low.",
]

predicted_labels = clusterer.predict(new_errors)
print(predicted_labels)  # Output might be ['database', 'system']
```

For additional information and advanced usage, please refer to the full documentation at <url documentation>.
