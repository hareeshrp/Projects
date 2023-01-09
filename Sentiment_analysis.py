# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 16:53:40 2023

@author: Hareesh
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
from bs4 import BeautifulSoup
import re
import numpy as np
import pandas as pd

#Using BERT
tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

#Webscrapping
url = 'https://www.yelp.com/biz/social-brew-cafe-pyrmont'
final_rev = []

for i in range(0,50,10):
    r = requests.get(url +'?start='+ str(i))
    soup = BeautifulSoup(r.text, 'html.parser')
    regex = re.compile('.*comment.*')
    results = soup.find_all('p', {'class':regex})
    reviews = [result.text for result in results]
    final_rev += reviews
    
#Converting to dataframe for modification
df = pd.DataFrame(np.array(final_rev), columns=['review'])

def sentiment_score(review):
    tokens = tokenizer.encode(review, return_tensors='pt')
    result = model(tokens)
    return int(torch.argmax(result.logits))+1

df['sentiment'] = df['review'].apply(lambda x: sentiment_score(x[:512]))

print(df)
    
