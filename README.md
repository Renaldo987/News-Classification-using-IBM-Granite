# News-Classification-using-IBM-Granite

**Title project** 

News Classification using IBM Granite



**Dataset Model**

Fake News detection dataset

Dataset separated in two files:
- Fake.csv (23502 fake news article)
- True.csv (21417 true news article)

Dataset columns:
- Title: title of news article
- Text: body text of news article
- Subject: subject of news article
- Date: publish date of news article

Dataset Link:
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset



**Project overview**

Background

In the digital information age, online news has become one of the primary sources of information for the public. However, the rapid spread of information is also accompanied by the rise of fake news, which can mislead readers, damage reputations, and even create social or political instability. A notable example is political demonstrations in Indonesia, where “buzzers” actively disseminated misleading or negative news to influence public opinion, fueling further conflict.
This phenomenon highlights the urgent need for an automated solution capable of distinguishing real news from fake news. Leveraging large language models such as IBM Granite, which has advanced natural language understanding, provides a strong opportunity to build an accurate, efficient, and scalable news classification system.

Problem Statement

The specific challenges addressed in this project include:
1. High volume of online news, making manual fact-checking impractical.
2. Similarity in linguistic patterns between fake and real news, which makes detection challenging.
3. Limitations of traditional machine learning models that rely only on surface-level features, failing to capture deeper semantic context.
4. Need for integration with modern AI models, enabling zero-shot or few-shot classification without extensive retraining.

Project Objectives

The project aims to:
1. Develop a news classification system capable of distinguishing real and fake news.
2. Evaluate the effectiveness of IBM Granite in performing zero-shot classification for news articles.
3. Provide benchmark comparisons with traditional models (Logistic Regression, SVM, Random Forest, etc.).
4. Deliver a framework that can be further extended into real-world applications, such as fact-checking platforms, media monitoring systems, and disinformation detection tools.

Approach

The project is carried out through the following systematic steps:
1. Data Collection : Using publicly available datasets (True.csv and Fake.csv), each labeled as real (1) or fake (0).
2. Data Preprocessing :
   - Removing duplicates and irrelevant columns (e.g., subject, date).
   - Cleaning text by removing HTML tags, special characters, and URLs.
   - Applying tokenization, stopword removal, stemming, and lemmatization.
   - Combining title + text into a single content field for richer input.
3. Exploratory Data Analysis (EDA):
   - Visualizing class distribution (fake vs. real news).
   - Checking missing values and ensuring data balance.
4. Modeling with IBM Granite :
   - Implementing zero-shot classification using Granite-3.3-8B-Instruct.
   - Creating a prompt-based approach: “Classify the following news strictly as either ‘True’ or ‘Fake’.”
   - Mapping responses to binary labels (1 = Real, 0 = Fake).
5. Model Evaluation :
   - Using metrics: accuracy, precision, recall, and F1-score.
   - Generating classification reports and confusion matrices for performance analysis.
6. Deployment Potential : Preparing the framework for integration into a prototype application that allows users to input articles and receive automated predictions.


**Insight & findings**
1. Overall High Performance
   - IBM Granite zero-shot classifier achieved 92.5% accuracy, showing strong ability to distinguish real vs. fake news without additional training.
   - Confusion Matrix: 20 Fake & 42 Real classified correctly, with only 5 misclassifications (2 False Negatives, 3 False Positives).
   - This confirms Granite’s strong generalization to unseen news articles.
2. Strong Precision on Real News
   - Precision (Real) = 0.95, meaning Granite rarely mislabels fake news as real.
   - Critical for minimizing the risk of spreading misinformation.
3. Balanced Recall Across Classes
   - Recall: Fake = 0.91 | Real = 0.93 → consistent detection performance across both classes.
   - Shows the model is not biased toward the more frequent “Real” class.
4. F1-Score Shows Robustness
   - Overall F1 = 0.94, reflecting a strong balance between precision and recall.
   - Ensures reliability across multiple evaluation perspectives, not just accuracy.
5. Class-Specific Observations
   - Fake news F1 = 0.89 vs. Real news F1 = 0.94.
   - Fake news remains harder to detect due to sensational or ambiguous writing styles.
6. Key Finding
   - Even without fine-tuning, IBM Granite shows robust zero-shot capabilities in fake news detection.
   - Improvement Potential: domain-specific fine-tuning or ensemble methods could further reduce errors, especially in fake news classification.


**AI support explanation**
In this project, Artificial Intelligence (AI) plays a central role in supporting the classification of news articles. The AI is applied through the following approaches:
1. Large Language Model (LLM) for Classification
   - IBM Granite is used as a zero-shot classifier, directly labeling news as True or Fake without additional training.
   - Its advanced natural language understanding allows the model to capture context, semantics, and subtle linguistic cues often present in fake news.
2. Text Preprocessing with NLP Techniques
   - AI-driven Natural Language Processing (NLP) methods such as tokenization, stopword removal, stemming, and lemmatization are applied to clean and normalize text.
   - This ensures the model focuses on meaningful content rather than irrelevant tokens.
3. Evaluation and Error Analysis
   - AI metrics (accuracy, precision, recall, F1-score) are used to evaluate Granite’s performance.
   - Misclassified cases are analyzed to understand the linguistic patterns that the model struggles with, guiding improvements for future iterations.
4. Potential Extensions of AI Use
   - Beyond classification, Granite or other LLMs could be extended to summarize news articles, analyze sentiment, or highlight potential bias in reporting.
   - This would create a more comprehensive AI-powered toolkit for combating misinformation.



