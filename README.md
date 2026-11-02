# Machine Learning Projects

Hi, I'm Datonte. I have a BSc in Computer Science and I'm studying for an MSc in FinTech. These are three machine learning projects I built while learning data science.

## Projects

### 1. Bank Customer Churn
**Notebook:** [Bank_Churn_Logistic_Regression.ipynb](Bank_Churn_Logistic_Regression.ipynb)

Predicts which bank customers are likely to leave the bank.

- **Data:** [Bank Customers (Kaggle)](https://www.kaggle.com/datasets/santoshd3/bank-customers), 10,000 customers
- **Model:** Logistic regression
- **Result:** 81% accuracy on the test set
- **What I learned:** only about 1 in 5 customers leave, so accuracy alone can be misleading. My model found only 19% of the customers who left (recall). Next I want to fix this with class weights and other models.

### 2. Spam Email Detector
**Notebook:** [spam_email_detector.ipynb](spam_email_detector.ipynb)

Sorts emails into spam and not spam. The same idea can be used to spot phishing and fraud messages.

- **Data:** [Spam Mails Dataset (Kaggle)](https://www.kaggle.com/datasets/venky73/spam-mails-dataset), 5,171 emails
- **Method:** TF-IDF to turn text into numbers, then three models compared: decision tree, SVM and logistic regression
- **Result:** logistic regression did best, with 98% accuracy on the test set
- **Extra:** the model is saved so it can be reused

### 3. Malaria Detection with a CNN
**Notebook:** [malaria_Parasite_Detection_Using_CNN.ipynb](malaria_Parasite_Detection_Using_CNN.ipynb)

Looks at photos of blood cells and tells if they are infected with malaria.

- **Data:** [Cell Images for Detecting Malaria (Kaggle)](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria), 27,558 images
- **Model:** a convolutional neural network (CNN) built with TensorFlow/Keras, with early stopping
- **Result:** 94.7% accuracy and 0.98 ROC-AUC on the test set
- **Extra:** the model is saved, and there is a simple upload box to test it on a new image

## Skills shown
- Python: pandas, NumPy, Matplotlib, Seaborn
- Machine learning with scikit-learn
- Deep learning with TensorFlow/Keras
- Data cleaning, data visualisation and model evaluation (accuracy, precision, recall, F1, ROC-AUC)

## How to run
The notebooks were made in Google Colab. The easiest way to run them is to open them in [Google Colab](https://colab.research.google.com/) and upload the dataset from the Kaggle link.

To run them on your own computer:
```bash
pip install -r requirements.txt
jupyter notebook
```
The datasets are not included in this repo. Download them from the Kaggle links above.
