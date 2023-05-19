# twitter_sentiment_analysis
Sentiment analysis on tweets using NLTK and ML classification models - Bernoulli Naive bayes, LinearSVC and Logistic Regression.


## Introduction

what is NLP(Natural language processing), it basically refers to that branch of AI which gives the computer the ability to understand and process text similar to us humans.
At this modern age, a large amount of unstructured data is generated everyday and it needs processing to gain insights. Some examples are content on social media, search history and even news articles. 

The process of analyzing natural language and making sense out of it falls under the field of Natural Language Processing (NLP). Sentiment analysis is a common NLP task, which involves classifying texts or parts of texts into a pre-defined sentiment, so declaring whether the text is having a positive sentiment or neutral or negative. You will use the Natural Language Toolkit (NLTK), a commonly used NLP library in Python, to analyze textual data.

So we shall create a model which processes tweets and predicts the sentiment.


## Importing Dataset

The dataset being used is the **sentiment140 dataset** from kaggle. It contains 1,600,000 tweets extracted using the Twitter API. The tweets have been annotated (0 = Negative, 4 = Positive) and they can be used to detect sentiment.
It contains 6 fields but we only require 'sentiment' and 'text' field.


## Data Preprocessing

This will reduce noise in the data, most of the preprocessing is done using python regex. Taking care of url, usernames, emojis, lower case and removing stopwords.

Once after tokenization we will lemmatize each word from a processed tweet and save them in a new python list.


## Data Analysis

We can use wordcloud to display negative and positive words which have been formatted based on frequency of occurance. I haven't used it here.


## TF-IDF

Term frequency-inverse document frequency is a text vectorizer that transforms the text into a usable vector. It combines 2 concepts, Term Frequency (TF) and Document Frequency (DF).

The term frequency is the number of occurrences of a specific term in a document. Term frequency indicates how important a specific term in a document. Term frequency represents every text from the data as a matrix whose rows are the number of documents and columns are the number of distinct terms throughout all documents.

Basically this text vectorizer gives weight to the tokens having repeated less than the ones having more frequency.
```python
from sklearn.feature_extraction.text import TfidfVectorizer as tfidfv

vectoriser = tfidfv(ngram_range=(1,2), max_features=500000)
vectoriser.fit(tX)
```


## Creating and Evaluating models

We're creating 3 different types of model for our sentiment analysis :

1. Bernoulli Naive Bayes (BernoulliNB)
2. Linear Support Vector Classification (LinearSVC)
3. Logistic Regression (LR)

As our dataset is not skewed, we shall be using accuracy as our evaluation metric.

We can see that Logistic Regression model performed better with an accuracy of nearly 82%.


## Using pickle

ML models can be saved and used by saving them as a pickle file. Later on when required we can load the models and predict.
I have demonstrated this using another file 'using_model'.

```python
from ipynb.fs.full.twitter_sentiment_analysis import load_models, predict

text = ['why is he always doing the same thing?', 'i have never seen anyone run that fast']
v, lr = load_models()

df = predict(v, lr, text)
df.head()
```

Output:

text	sentiment

why is he always doing the same thing?	Negative

i have never seen anyone run that fast	Positive
