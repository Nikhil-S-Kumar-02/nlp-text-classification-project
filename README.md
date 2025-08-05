# NLP Text Classification Project

This repository contains two NLP projects built as part of a Data Science course:

1. **IMDb Sentiment Analysis (Part A)**  
   A binary classification task to predict whether a movie review is Positive or Negative.

2. **News Category Classification (Part B)**  
   A multi-class classification task to predict the category of news articles (e.g., politics, sports, technology).

---

##  Project Structure

```
nlp-text-classification-project/
│
├── Part_A_IMDb_Sentiment_Analysis.ipynb
├── Part_B_News_Category_Classification.ipynb
├── Part_A_Report.pdf
├── Part_B_Report.pdf
├── README.md
```

---

##  Part A: IMDb Sentiment Analysis

- **Dataset:** 50,000 labeled reviews (positive/negative)
- **Preprocessing:** HTML removal, lowercasing, punctuation stripping, stopword removal, lemmatization
- **Feature Extraction:** TF-IDF (Top 5000 terms)
- **Model:** Logistic Regression
- **Accuracy:** 88.5%

**Key Libraries:** `NLTK`, `scikit-learn`, `BeautifulSoup`, `pandas`, `matplotlib`

---

##  Part B: News Category Classification

- **Dataset:** News articles labeled with categories (politics, sports, tech, etc.)
- **Preprocessing:** Tokenization, cleaning, vectorization
- **Feature Extraction:** TF-IDF, Bag-of-Words
- **Models Used:** Logistic Regression, Naive Bayes
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score

---

##  Learnings

- Practical implementation of NLP pipelines from scratch
- Text vectorization with TF-IDF
- Binary and multi-class text classification
- Interpreting classification metrics

---

##  Tools & Technologies

- Python
- Jupyter Notebook
- Scikit-learn
- Pandas
- NLTK
- Matplotlib / Seaborn

---

> This project was done as part of a Data Science course assignment for educational and demonstrative purposes.
