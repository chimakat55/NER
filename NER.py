# -*- coding: utf-8 -*-

# Part-of-Speech Tagging and Named Entity Recognition


import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger') # 94.0% accuracy

def preprocess(sentence):
    tokens = nltk.word_tokenize(sentence)
    tokens_pos = nltk.pos_tag(tokens)
    return tokens_pos

!pip install html5lib # Library for processing HTML
!pip install BeautifulSoup4 # Library for web scraping

from bs4 import BeautifulSoup
import requests
import re

blacklist = [ # List with elements we don't want to keep
    'script',
    'style',
    'aside'
]

def url_to_string(url):
    request = requests.get(url)
    html = request.text
    soup = BeautifulSoup(html, "html5lib")
    
    for script in soup(blacklist):
        script.extract()
    text = [paragraph.get_text() for paragraph in soup.find_all('p')]
    return " ".join(text)

## Pick an article to NER ##

url = "https://www.lemonde.fr/politique/article/2022/02/01/presidentielle-2022-le-parti-socialiste-au-bord-de-la-crise-de-nerfs_6111818_823448.html"
article = url_to_string(url)
print(article)

## Get french spacy ##

!python -m spacy download fr_core_news_sm
import fr_core_news_sm 

nlp = fr_core_news_sm.load() 
nlp

article_lower = article.lower()
article_doc_lower = nlp(article_lower)

print(len(article_doc_lower.ents))

article_doc = nlp(article)
type(article_doc)

print(len(article_doc.ents))
print(article_doc.ents)

labels = [x.label_ for x in article_doc.ents]
print(labels)

from collections import Counter
labels_dict = Counter(labels) 
print(labels_dict)

entities = [x.text for x in article_doc.ents]
entities
print(Counter(entities).most_common(10))

for label in labels_dict.keys():
  print(label)
  entities = [x.text for x in article_doc.ents if x.label_ == label]
  print(Counter(entities).most_common(5))

sentences = [x for x in article_doc.sents]
sentences
print(sentences[5])