# Machine Learning Projects

Hi, I'm Datonte. I have a BSc in Computer Science and I'm studying for an MSc in FinTech. These are the machine learning projects I built while learning data science. The first three are my main projects.

## Main projects

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

## More projects

| Project | What it does | Main result |
|---|---|---|
| [Breast Cancer Classification (Neural Network)](Breast_Cancer_Classification_Using_ANN.ipynb) | Neural network (TensorFlow/Keras) that classifies tumours as malignant or benign from 30 features, using 569 patient records | 95% test accuracy |
| [Breast Cancer (Logistic Regression)](Breast_Cancer_Logistic_Regression.ipynb) | The same problem solved with logistic regression | 95.6% test accuracy, 95.3% recall |
| [Heart Disease Prediction](Heart_disease_model.ipynb) | Compares logistic regression and random forest, using SMOTE for class balance and PCA | 86.7% test accuracy, 0.96 ROC-AUC |
| [Face Mask Detection](Face_mask_detection_model.ipynb) | CNN that tells whether a person in a photo is wearing a face mask, trained on 7,553 images | 93% test accuracy |
| [Boston House Prices](Boston_House_Prices_Linear_Regression.ipynb) | Linear regression to predict house prices | R² of 0.67 on the test set |
| [Machine Learning Practice](Customer_Churn_MachineLearning.ipynb) | Practice notebook where I tried regression and classification models (logistic regression, decision trees, random forest, XGBoost, LightGBM, CatBoost) on several datasets | Practice work |

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
