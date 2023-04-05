import csv
import io
import json
import numpy as np
import os
import pandas as pd
import random
import requests
import spacy
import spacy_transformers
import sys
import time
import torch

from keybert import KeyBERT
from keyphrase_vectorizers import KeyphraseCountVectorizer, KeyphraseTfidfVectorizer
from numpy import NaN
from os.path import exists as file_exists
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForPreTraining, TrainingArguments, Trainer


def get_raw_data(path):
    columns = [
        'id',
        'Titel',
        'Marketingtext',
        'Markenname',
        'Funktionsbezeichnung',
        'Rechtliche Bezeichnung',
        'Herkunftsland']
   
    df = pd.read_excel(path, header=None, names=columns, usecols=[0,1,2,5,6,8,9], skiprows=1, dtype={"id": np.int64})   # read product data with custom column names
    df = df[df.id.notnull() & df.Titel.notnull() & df.Marketingtext.notnull()]                                          # remove rows with empty fields
    df = df[df["Marketingtext"].apply(lambda x: len(x) > 100)]                                                          # remove short texts
    df = df.drop_duplicates(subset="id", keep="first")                                                                  # remove duplictates
    df["Marketingtext"] = df["Marketingtext"].str.replace("\n", " ")                                                    # replace false imported tap chars

    print(f"Anzahl Zeilen: {len(df)}")
    print(df.dtypes)
    df.head(2)

    return df


def create_keywords_bert(d, cuda_avbl):

    print("start create_keywords_bert")

    if cuda_avbl:
        spacy.require_gpu()
        pipe = "de_dep_news_trf"
    else:
        pipe = "de_core_news_lg"

    kw_model = KeyBERT(model=SentenceTransformer("distiluse-base-multilingual-cased-v1"))                           # init KeyBert with multilingual model
    #vectorizer = KeyphraseCountVectorizer(spacy_pipeline=pipe, pos_pattern='<ADJ.*>*<N.*>+', stop_words="german")
    vectorizer = KeyphraseTfidfVectorizer(spacy_pipeline=pipe, pos_pattern='<ADJ.*>*<N.*>+', stop_words="german")   # define german keyphrase vectorizer

    counter = 0
    kws_set = set()

    for key, value in d.items():
        kws_tupels = kw_model.extract_keywords(docs=value[1], vectorizer=vectorizer)                                # get keywords from text with defined model
        kws_list = []
        for word in kws_tupels:                                                                                     # extract keywords from returned object
            kws_list.append(word[0])                                                                                # collect keywords in list
            kws_set.add(word[0])                                                                                
        d[key].append(kws_list)                                                                                     # append list of keywords to related text based on id
        
        if counter % 500 == 0:
            print(f"{counter} / {len(d)} processed")
        counter = counter + 1

    print(f"Number of unique keywords: {len(kws_set) :,}")

    return d


def split_data(data, S):
    # Shuffle ids
    ids = list(data.keys())
    random.shuffle(ids)

    # Split into training and validation sets    
    train_size = int(S * len(data))

    train_ids = ids[:train_size]
    val_ids = ids[train_size:]

    train_data = dict()
    for id in train_ids:
        train_data[id] = data[id]

    val_data = dict()
    for id in val_ids:
        val_data[id] = data[id]

    return train_data, val_data


def get_tokenier(MODEL, special_tokens=None):
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if special_tokens:
        tokenizer.add_special_tokens(special_tokens)
        print("Special tokens added")
    return tokenizer
    

def get_model(MODEL, cuda_avbl, tokenizer, special_tokens=None, load_model_path=None):

    if special_tokens:
        config = AutoConfig.from_pretrained(MODEL, 
                                            bos_token_id=tokenizer.bos_token_id,
                                            eos_token_id=tokenizer.eos_token_id,
                                            sep_token_id=tokenizer.sep_token_id,
                                            pad_token_id=tokenizer.pad_token_id,
                                            output_hidden_states=False)
    else: 
        config = AutoConfig.from_pretrained(MODEL,                                     
                                            pad_token_id=tokenizer.eos_token_id,
                                            output_hidden_states=False)    

    
    model = AutoModelForPreTraining.from_pretrained(MODEL, config=config)

    # special tokens added, model needs to be resized accordingly
    if special_tokens:
        model.resize_token_embeddings(len(tokenizer))

    # load if model path is given
    if load_model_path:
        if cuda_avbl:
            model.load_state_dict(torch.load(load_model_path))
        else:
            model.load_state_dict(torch.load(load_model_path, map_location=torch.device('cpu'))) 

    if cuda_avbl:
        model.cuda()
        
    return model
