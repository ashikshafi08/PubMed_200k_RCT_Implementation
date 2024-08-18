# PubMed 200k RCT Implementation

This repository contains the implementation of a deep learning model based on the 2017 paper "PubMed 200k RCT: A Dataset for Sequential Sentence Classification in Medical Abstracts." The goal is to explore the ability of NLP models to classify sentences within abstracts, identifying the role each sentence plays in the overall structure.

## Project Overview

The purpose of this project is to build a deep learning model capable of classifying sentences in medical abstracts into categories such as "BACKGROUND," "OBJECTIVE," "METHODS," "RESULTS," and "CONCLUSIONS." The model will take unstructured text from research abstracts and predict the section label for each sentence, making the abstracts easier to read and navigate.

### Dataset

The dataset used in this project is the PubMed 200k RCT dataset, which consists of research abstracts from randomized controlled trials (RCTs). Each abstract is labeled at the sentence level, indicating the role of each sentence within the abstract.

### Model Architecture

The project experiments with several model architectures, including:

1. **Baseline Model (Model 0):** TF-IDF with Multinomial Naive Bayes.
2. **Model 1:** A hybrid model combining token embeddings, character embeddings, and positional embeddings, implemented using a Bidirectional LSTM network.

### Implementation Details

The implementation involves several key steps:

1. **Data Preprocessing:**
   - Downloading and reading the dataset.
   - Splitting the text into sentences and characters.
   - Encoding the labels using OneHotEncoder and LabelEncoder.
   - Vectorizing the text and character sequences.

2. **Modeling:**
   - Building and training the baseline model using TF-IDF and Naive Bayes.
   - Constructing the hybrid deep learning model with token, character, and positional embeddings.
   - Training the models and evaluating their performance on validation data.

3. **Evaluation:**
   - Evaluating the models using metrics such as accuracy, precision, recall, and F1 score.
   - Comparing the performance of different models and configurations.

### Results

The best-performing model (Model 1) achieved an accuracy of approximately 87.31% on the validation dataset. The model combines token-level and character-level embeddings with positional information, yielding robust performance across different sections of the abstracts.

### Usage

#### Prerequisites

- Python 3.7+
- TensorFlow
- Scikit-learn
- Pandas
- Matplotlib

#### Running the Code

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Franck-Dernoncourt/pubmed-rct.git
   cd pubmed-rct
   ```

### Run the notebook:

Open the provided Jupyter notebook (`PubMed_200k_RCT_Implementation.ipynb`) in your favorite editor (Jupyter Notebook, Google Colab, etc.) and follow the steps to train and evaluate the models.

### Train the Model:

The notebook will guide you through the process of training the model, including data preprocessing, building the model, and evaluating its performance.

### Evaluate the Model:

After training, you can evaluate the model on the test set and visualize the results.

### Model Summary

Here’s a summary of the best model (Model 1):

#### Inputs:

- Tokenized sentences
- Character sequences
- Positional information (line number, total lines)

#### Outputs:

- Sentence classification into one of the five categories: BACKGROUND, OBJECTIVE, METHODS, RESULTS, CONCLUSIONS.

#### Layers:

- Token and character embedding layers.
- Bidirectional LSTM layers for sequence processing.
- Dense layers for positional information.
- Output layer with softmax activation for classification.

### Repository Structure

- **`data/`**: Contains the dataset files (train, validation, test).
- **`notebooks/`**: Jupyter notebook with the full implementation.
- **`models/`**: Saved models after training.
- **`README.md`**: This file.

### Future Work

Potential improvements and extensions for this project include:

- Experimenting with different types of embeddings (e.g., BERT, GloVe).
- Fine-tuning the hyperparameters for better performance.
- Extending the model to handle abstracts with more complex structures.

### License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

### Acknowledgements

- The original dataset and paper: "PubMed 200k RCT: a Dataset for Sequential Sentence Classification in Medical Abstracts."
- The deep learning community for providing open-source tools and libraries.
