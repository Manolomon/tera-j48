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

# %%
import numpy as np
import re
import nltk
import pandas as pd
nltk.download('stopwords')
import pickle
from nltk.corpus import stopwords

# %%
data = pd.read_csv('nfr.csv')
data = data.rename(columns={'RequirementText': 'text', 'class': 'label'})
data
X = data.text.to_numpy()
y = data.label.to_numpy()

# %%
