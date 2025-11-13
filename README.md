# 🔤 Language-Prediction-Using-Multinominal-Naive-Bayes

This repository contains the implementation of my project titled _"COVID-19 Exploratory Data Analysis and Forecasting"_, at the National Centre for Artificial Intelligence and Robotics (NCAIR), Nigeria.

---

## 🧭 Project Overview
This project focused on building a model using multinominal naive bias machine learning algorithm that could predict languages such as English, French, Spanish and German.

---
## 🧩 Project Objectives

- To develop a model that can predict languages specifically `English`, `German`, `French` and `Spanish`.
- To perform a language distrbution analysis of the dataset.

---

## 🧰 Tools and Technologies Used

| **Tool/Library** | **Purpose in the Project** |
|------------------|----------------------------|
| **Python** | Served as the main programming language for implementing the language prediction model and data analysis. |
| **Pandas** | Used for loading, cleaning, and manipulating the dataset. |
| **NumPy** | Assisted with numerical operations and array handling. |
| **Scikit-learn (sklearn)** | Provided various machine learning utilities including model training, evaluation metrics, and data preprocessing. |
| **TfidfVectorizer** | Used for converting text data into numerical vector representations suitable for model training. |
| **LabelEncoder** | Applied to encode categorical language labels into numerical form. |
| **Train-Test Split** | Used to divide the dataset into training and testing subsets for performance evaluation. |
| **Multinomial Naive Bayes (MultinomialNB)** | Implemented as the core classification algorithm for predicting the language of a given text input. |
| **GridSearchCV** | Used for hyperparameter tuning to improve model accuracy. |
| **PowerBI** | Utilized for visualizing the data distribution across languages. |
| **Joblib** | Used to save and load the trained machine learning. |

---
## 🔬 Step-by-Step Procedure

1. **Data loading:**  
   The language dataset was loaded into a `pandas` DataFrame from a CSV file.

2. **Language filtering:**  
   The dataset was filtered to keep only the target languages (e.g., English, German, French, Spanish) and any unwanted rows or columns were removed.

3. **Text cleaning & preprocessing:**  
   Text entries were normalized (lowercased), and minimal cleaning steps were applied (trimming whitespace, optional punctuation removal) to prepare raw text for vectorization.

4. **Label encoding:**  
   Language labels were converted from string categories to integer codes using `LabelEncoder` so they could be used by the classifier.

5. **Feature extraction (TF–IDF):**  
   Text data were transformed into numeric feature vectors using `TfidfVectorizer`, converting raw text into a machine-readable form emphasizing important terms.

6. **Train/test split:**  
   The dataset was split into training and testing subsets using `train_test_split` to allow unbiased evaluation of the model on unseen data.

7. **Model selection (MultinomialNB):**  
   The **Multinomial Naive Bayes** algorithm was chosen as the classifier because it suits discrete feature counts and TF–IDF representations for text classification.

8. **Hyperparameter tuning (GridSearchCV):**  
   `GridSearchCV` was run on the training data to find the best hyperparameters (for exampl, smoothing `alpha`), optimizing performance via cross-validation.

9. **Model training:**  
   The best estimator returned by the grid search was fitted on the training data to learn the mapping from TF–IDF features to language labels.

10. **Prediction & evaluation:**  
    The trained model predicted labels on the test set. Performance was measured using metrics such as accuracy, precision, recall, F1-score, and a confusion matrix to inspect per-class performance.

11. **Visualization of results:**  
    The PowerBI application was used to visualize the language distributition from the dataset.

12. **Model Saving:**  
    The final trained model was saved to disk using `joblib` (or `pickle`) for later reuse in inference.

---
## 📈 Evaluation Metrics

| **Metric**      | **Value** | **Remarks** |
|------------------|-----------|-------------|
| **Accuracy**     | 0.9955 | The model achieved high accuracy, correctly predicting almost all language samples in the dataset. This reflects strong learning ability but could indicate overfitting due to near-perfect results. |
| **Precision**    | 0.9955 | The high precision value showed that the model rarely misclassified text samples into the wrong language, demonstrating excellent discrimination between linguistic patterns. |
| **Recall**       | 0.9955 | The recall score indicated that the model successfully identified nearly all true language instances with minimal false negatives, confirming its effectiveness on the dataset. |

<p align="center">
    <img src="Visualization of the Language Distribution on PowerBI.png" alt="Visualization of the Language Distribution on PowerBI" width="1500"/>
    <br>
    <em> Visualization of the Language Distribution on PowerBI</em>
</p>

---
## ⚠️ Disclaimer

While the evaluation metrics indicate exceptional model performance, the consistency and magnitude of these results suggest potential **overfitting** meaning the model might perform less effectively on unseen data. Future work could focus on testing the model using **real-world multilingual datasets** to validate its generalization ability, or explore **more advanced approaches such as transformer-based models or Large Language Models (LLMs)** to improve adaptability and contextual understanding in language prediction tasks.

---

## 📌 Note

Please kindly note that this README file is a summarized version of the full implementation of this project. The complete implementation can be accessed via the [program script](Language_Prediction_Using_Multinominal_Naive_Bayes.ipynb) and analysis [visualization](Language-Distribution-Analysis-(PowerBI-Visualization).pbix).

---
