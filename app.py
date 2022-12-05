# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:15:42 2022

@author: amand
"""

import streamlit as st
import pandas as pd
import pickle as pkl
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from collections import Counter
from heapq import nlargest
import re
import os
from spacy import displacy
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize,sent_tokenize
nltk.download('stopwords')

from nltk.corpus import stopwords

nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import scipy.spatial
import torch


with open("santorini_df.pkl", "rb") as file1: 
    df = pkl.load(file1)
with open("santorini_corpus.pkl", "rb") as file2: 
    corpus = pkl.load(file2)
with open("santorini_embeddings.pkl", "rb") as file3: 
    corpus_embeddings = pkl.load(file3)
    
embedder = SentenceTransformer('all-MiniLM-L6-v2')
    

queries = st.text_input('What are you looking for in a hotel?', value = "",
                        type=str) 

st.write('You are looking for a hotel that is', queries)

top_k = min(5, len(corpus))
for query in queries:
    query_embedding = embedder.encode(query, convert_to_tensor=True)

    # We use cosine-similarity and torch.topk to find the highest 5 scores
    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)
    print("\n\n======================\n\n")
    print("\nTop 5 Hotel reviews matching your description:")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    