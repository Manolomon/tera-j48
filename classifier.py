# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

#hgzdfdfdhjgfdhgf

# %%
import numpy as np
import re
import nltk
import pandas as pd
nltk.download('stopwords')
import pickle
from nltk.corpus import stopwords
import weka.core.jvm as jvm
from weka.core.dataset import create_instances_from_matrices

# %%
data = pd.read_csv('nfr.csv')
data = data.rename(columns={'RequirementText': 'text', 'class': 'label'})
data

# %%
interest = ['US', 'SE', 'PE', 'O']
data.label.unique()

# %%
data = data.loc[data['label'].isin(interest)]
X = data.text.to_numpy()
y = data.label.to_numpy(dtype='<U6')
data

# %%
documents = []

from nltk.stem import WordNetLemmatizer

stemmer = WordNetLemmatizer()

for sen in range(0, len(X)):
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(X[sen]))
    
    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    
    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
    
    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    
    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)
    
    # Converting to Lowercase
    document = document.lower()
    
    # Lemmatization
    document = document.split()

    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)
    
    documents.append(document)

# %%
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(documents).toarray()

# %%
from sklearn.feature_extraction.text import TfidfTransformer
tfidfconverter = TfidfTransformer()
X = tfidfconverter.fit_transform(X).toarray()

# %%
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(y)
y = le.transform(y)

# %%
jvm.start()

# %%
from weka.filters import Filter
dataset = create_instances_from_matrices(X, y, name="Quality Attributes")
nominal = Filter(classname="weka.filters.unsupervised.attribute.NumericToNominal", options=["-R", "last"])
nominal.inputformat(dataset)
nominaldata = nominal.filter(dataset)
nominaldata.class_is_last()

# %%
from weka.classifiers import Classifier, Evaluation
from weka.core.classes import Random
cls = Classifier(classname="weka.classifiers.trees.J48", options=["-C", "0.3"])
cls.build_classifier(nominaldata)

print(cls)

import weka.plot.graph as graph  # NB: pygraphviz and PIL are required
graph.plot_dot_graph(cls.graph)

evaluation = Evaluation(nominaldata)                     # initialize with priors
evaluation.crossvalidate_model(cls, nominaldata, 10, Random(42))  # 10-fold CV
print(evaluation.summary())
print("pctCorrect: " + str(evaluation.percent_correct))
print("incorrect: " + str(evaluation.incorrect))

# %%
jvm.stop()

# %%
