import requests
 
# get URL

# url = input("Enter url: ")
# page = requests.get(url)
 
# display status code
# print(page.status_code)
 
# display scraped data
# print(page.content)
# import required modules
from bs4 import BeautifulSoup
import requests
 
# get URL
#url = input("Enter url: ")
# page = requests.get(url)
 
# # scrape webpage
# soup = BeautifulSoup(page.content, 'html.parser')
 
# # display scraped data
# html_text = soup.prettify()
# print(html_text)
# import re
# re_html = re.compile(r'<[^>]+>')
# re_html.sub('',html_text)
import pandas as pd
import string
df=input('give string')
string.punctuation
def remove_punctuations(text):
    punctuations = string.punctuation
    return text.translate(str.maketrans('', '', punctuations))
df = remove_punctuations(df)
#df
### Removal of stopwords

#Stop word removal is one of the most commonly used preprocessing steps across different NLP applications. The idea is simply removing the words that occur commonly across all the documents in the corpus. 
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
sample_stopwords = stopwords.words('english')
print(sample_stopwords)
def remove_stopwords(text):
    return " ".join([word for word in text.split() if word not in sample_stopwords])
df = remove_stopwords(df)
#df
### Removal of frequent words
from collections import Counter

word_count = Counter()
for text in df:
    for word in text.split():
        word_count[word] += 1
        
word_count.most_common(10)
        
frequent_words = set(word for (word,wc) in word_count.most_common(3))
def remove_freq_words(text):
    
    return " ".join([word for word in text.split() if word not in frequent_words])
df = remove_freq_words(df)
#df
### Removal of Rare words
rare_words = set(word for (word,wc) in word_count.most_common()[:-10:-1])
rare_words
def remove_rare_words(text):
    
    return " ".join([word for word in text.split() if word not in rare_words])
df = remove_rare_words(df)
#df
### Removal of Special Characters
import re

def remove_spl_characters(text):
    
    text = re.sub('[^a-zA-Z0-9]', ' ', text)
    text = re.sub('\s+', ' ',text)
    
    return text
df = remove_spl_characters(df)
#df
### Stemming

#Stemming is the process of reducing a word to its stem that affixes to suffixes and prefixes or to the roots of words known as "lemmas".
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
def stemming(text):
    return " ".join([stemmer.stem(word) for word in text.split()])
          
df = stemming(df)
#df
### Lemmatization

#Lemmatization is a text normalization technique used in Natural Language Processing (NLP), that switches any kind of a word to its base root mode. Lemmatization is responsible for grouping different inflected forms of words into the root form, having the same meaning.
import nltk
nltk.download('omw-1.4')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

wordnet_map = {"N": wordnet.NOUN , "V" : wordnet.VERB , "J" : wordnet.ADJ , "A" : wordnet.ADV}

def lemmatize(text):
    pos_text = pos_tag(text.split())
    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word,pos in pos_text])
df = lemmatize(df)
#df
import torch
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model1 = RobertaModel.from_pretrained("microsoft/codebert-base")
model1.to(device)

import pickle
# create an iterator object with write permission - model.pkl
with open('model1.pkl', 'wb') as files:
     pickle.dump(model1, files)