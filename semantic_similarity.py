import sys
import pandas as pd
import spacy
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet as wn
import spacy
nlp = spacy.load("en_core_web_sm")
from nltk.corpus import stopwords
nltk.download("stopwords")
from nltk.stem import WordNetLemmatizer 
from collections import defaultdict
import pprint
import argparse
import difflib
from googletrans import Translator
import numpy as np
import pickle
translator = Translator()
import time
from nltk.corpus import wordnet_ic
from gensim.models import KeyedVectors
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import cross_val_score
from nltk.corpus import wordnet as wn
brown_ic = wordnet_ic.ic('ic-brown.dat')
nltk.download('wordnet_ic')
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn
import gensim
import os
import collections
import smart_open
import re
import random
from collections import Counter
import spotlight
nltk.download('wordnet')
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords
from sklearn.svm import SVR, LinearSVR
# Evaluation
from sklearn.metrics import accuracy_score, precision_score, classification_report, confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

stop_words = stopwords.words('english')
pd.set_option('display.max_columns', None)  

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = WordNetLemmatizer()
vec1 = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)
vec2 = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)
stop = stopwords.words('english')


#lemmatization
def lemmatize(s):
  
    s = [lemmatizer.lemmatize(word) for word in s]
    return s

def dependency_parse(sentence):
    doc=nlp(sentence)
    for token in doc:    
        if(token.dep_=="ROOT"):
            return lemmatizer.lemmatize(token.text)
        
# =========== util func ==============
def penn_to_wn(tag):
    """ Convert between a Penn Treebank tag to a simplified Wordnet tag """
    if tag.startswith('N'): return 'n'
    if tag.startswith('V'): return 'v'
    if tag.startswith('J'): return 'a'
    if tag.startswith('R'): return 'r'
    return None
 
def tagged_to_synset(word, tag):
    wn_tag = penn_to_wn(tag)
    if wn_tag is None:
        return None
    try:
        return wn.synsets(word, wn_tag)[0]
    except:
        return None

notin_cnt = [0]
# =========== feature extraction ==============
def sentence_similarity_word_alignment(sentence1, sentence2):
    """ compute the sentence similarity using Wordnet and ppdb """
    # Tokenize and tag
    sentence1 = pos_tag(word_tokenize(sentence1))
    sentence2 = pos_tag(word_tokenize(sentence2)) 
    # Get the synsets for the tagged words
    synsets1 = [tagged_to_synset(*tagged_word) for tagged_word in sentence1]
    synsets2 = [tagged_to_synset(*tagged_word) for tagged_word in sentence2]
    # Filter out the Nones
    synsets1 = [ss for ss in synsets1 if ss]
    synsets2 = [ss for ss in synsets2 if ss]
    score, count = 0.0, 0
    ppdb_score, align_cnt = 0, 0
    # For each word in the first sentence
    for synset in synsets1:
        # Get the similarity value of the most similar word in the other sentence
        L = [synset.path_similarity(ss) for ss in synsets2]
        L_prime = L
        L = [l for l in L if l]

      

        if L: 
            best_score = max(L)
            score += best_score
            count += 1
    # Average the values
    if count >0: score /= count

    return score

def sentence_similarity_simple_baseline(s1, s2,counts = None):
    def embedding_count(s):
        ret_embedding = defaultdict(int)
        for w in s.split():
            w = w.strip('?.,')
            ret_embedding[w] += 1
        return ret_embedding
    first_sent_embedding = embedding_count(s1)
    second_sent_embedding = embedding_count(s2)
    Embedding1 = []
    Embedding2 = []
    if counts:
        for w in first_sent_embedding:
            Embedding1.append(first_sent_embedding[w] * 1.0/ (counts[w]+0.001))
            Embedding2.append(second_sent_embedding[w] *1.0/ (counts[w]+0.001))
    else:
        for w in first_sent_embedding:
            Embedding1.append(first_sent_embedding[w])
            Embedding2.append(second_sent_embedding[w])
    ret_score = 0
    if not 0 == sum(Embedding2): 
      
        sm= difflib.SequenceMatcher(None,Embedding1,Embedding2)
        ret_score = sm.ratio()*5 
    return ret_score

def extract_overlap_pen(s1, s2):
    """
    :param s1:
    :param s2:
    :return: overlap_pen score
    """
    ss1 = s1.strip().split()
    ss2 = s2.strip().split()
    ovlp_cnt = 0
    for w1 in ss1:
        ovlp_cnt += ss2.count(w1)
    score = 2 * ovlp_cnt / (len(ss1) + len(ss2) + .0)
    return score

def all_token_diff(s1,s2):
    s1, s2 = word_tokenize(s1), word_tokenize(s2)
    pos1, pos2 = pos_tag(s1), pos_tag(s2)
    # all tokens
    return  abs(len(s1) - len(s2)) / float(len(s1) + len(s2))

def adj_diff(s1,s2):
    s1, s2 = word_tokenize(s1), word_tokenize(s2)
    pos1, pos2 = pos_tag(s1), pos_tag(s2)
    # all tokens
    cnt1 = len([1 for item in pos1 if item[1].startswith('J')])
    cnt2 = len([1 for item in pos2 if item[1].startswith('J')])
    if cnt1 == 0 and cnt2 == 0:
        t2 = 0
    else:
        t2 = abs(cnt1 - cnt2) / float(cnt1 + cnt2)
        
    return t2

def adv_diff(s1,s2):
    s1, s2 = word_tokenize(s1), word_tokenize(s2)
    pos1, pos2 = pos_tag(s1), pos_tag(s2)
    cnt1 = len([1 for item in pos1 if item[1].startswith('R')])
    cnt2 = len([1 for item in pos2 if item[1].startswith('R')])
    if cnt1 == 0 and cnt2 == 0:
        t3 = 0
    else:
        t3 = abs(cnt1 - cnt2) / float(cnt1 + cnt2)
        
    return t3


def noun_diff(s1,s2):
    s1, s2 = word_tokenize(s1), word_tokenize(s2)
    pos1, pos2 = pos_tag(s1), pos_tag(s2)
    # all nouns
    cnt1 = len([1 for item in pos1 if item[1].startswith('N')])
    cnt2 = len([1 for item in pos2 if item[1].startswith('N')])
    if cnt1 == 0 and cnt2 == 0:
        t4 = 0
    else:
        t4 = abs(cnt1 - cnt2) / float(cnt1 + cnt2)
        
    return t4

def verb_diff(s1,s2):
    s1, s2 = word_tokenize(s1), word_tokenize(s2)
    pos1, pos2 = pos_tag(s1), pos_tag(s2)
    # all verbs
    cnt1 = len([1 for item in pos1 if item[1].startswith('V')])
    cnt2 = len([1 for item in pos2 if item[1].startswith('V')])
    if cnt1 == 0 and cnt2 == 0:
        t5 = 0
    else:
        t5 = abs(cnt1 - cnt2) / float(cnt1 + cnt2)
    return t5

def sentence_similarity_information_content(sentence1, sentence2):

    ''' compute the sentence similairty using information content from wordnet '''
    # Tokenize and tag
    sentence1 = pos_tag(word_tokenize(sentence1))
    sentence2 = pos_tag(word_tokenize(sentence2)) 
    # Get the synsets for the tagged words
    synsets1 = [tagged_to_synset(*tagged_word) for tagged_word in sentence1]
    synsets2 = [tagged_to_synset(*tagged_word) for tagged_word in sentence2]
    # Filter out the Nones
    synsets1 = [ss for ss in synsets1 if ss]
    synsets2 = [ss for ss in synsets2 if ss]
    score, count = 0.0, 0
    ppdb_score, align_cnt = 0, 0
    # For each word in the first sentence
    for synset in synsets1:
        L = []
        for ss in synsets2:
            try:
                L.append(synset.res_similarity(ss, brown_ic))
            except:
                continue
        if L: 
            best_score = max(L)
            score += best_score
            count += 1
    # Average the values
    if count >0: score /= count
    return score


def extract_res_vec_similarity(s1, s2):
    first_sents_embeddings = np.empty([0,300])
    second_sents_embeddings = np.empty([0,300])

    first_vecs = np.array([])
    for w in s1.split():
        w = w.strip('?.,')
        if w in vec1:
            first_vec = np.array([vec1[w]])
            if first_vecs.shape[0] == 0:
                first_vecs = first_vec
            else:
                first_vecs = np.vstack((first_vecs, first_vec))
        else:
            if first_vecs.shape[0] == 0:
                first_vecs = np.random.normal(0, 5, 300)
            else:
                first_vecs = np.vstack((first_vecs, np.random.normal(0, 5, 300)))
        # print("first ")
        # print(first_vecs.shape)
    if(first_vecs.shape == (300, )):
        temp = first_vecs
    else:
        temp = np.mean(first_vecs, axis=0)
    # print(temp.shape)
    first_sents_embeddings = np.append(first_sents_embeddings, [temp], axis=0)

    second_vecs = np.array([])  
    for w in s2.split():
        w = w.strip('?.,')
        if w in vec2:
            second_vec = np.array([vec2[w]])
            if second_vecs.shape[0] == 0:
                second_vecs = second_vec
            else:
                second_vecs = np.vstack((second_vecs, second_vec))
        else:
            if second_vecs.shape[0] == 0:
                second_vecs = np.random.normal(0, 5, 300)
            else:
                second_vecs = np.vstack((second_vecs, np.random.normal(0, 5, 300)))
        # print("second ")
        # print(second_vecs.shape)
    if(second_vecs.shape == (300,)):
        temp = second_vecs
    else:
        temp = np.mean(second_vecs, axis=0)
    # print(temp.shape)
    second_sents_embeddings = np.append(second_sents_embeddings, [temp], axis=0)

    for i in range(len(first_sents_embeddings)):
        # cosine similarity

        ret = np.dot(first_sents_embeddings[i], second_sents_embeddings[i]) / (np.linalg.norm(first_sents_embeddings[i]) * np.linalg.norm(second_sents_embeddings[i]))
        ret = 5*(ret + 1) / 2

    return ret


def extract_doc2vec_similarity(s1,s2, model):
    s1 = [w.strip('?.,') for w in s1.split()]
    s2 = [w.strip('?.,') for w in s2.split()]
    embed1 = model.infer_vector(s1)
    embed2 = model.infer_vector(s2)
    ret = np.dot(embed1,embed2)
    return ret



def duplicates(value1,value2):
    
    dups = Counter(value1) - Counter(value2)
    return len(dups)

def get_annotation(text):
   # print(text)
    try:
        annotations = spotlight.annotate('http://api.dbpedia-spotlight.org/en/annotate',text,
                              confidence=0.4, support=20)
    except Exception as e:
       # print(e)
        return {'URI':[],'types':[]}
    
    URI_lst=[]
    types_lst=[]
    
    for ann_dict in annotations:
        URI_lst.append(ann_dict['URI'])
        types_lst.append(ann_dict['types'])
    
    return {'URI':URI_lst,'types':types_lst}

def get_common_types(sent1_types,sent2_types):
    s1=set(sent1_types)
    s2=set(sent2_types)
    if len(s1.union(s2))>0:
        return len(s1.intersection(s2))/len(s1.union(s2))
    return 0.5



def get_common_entities(text1,text2):
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    ent1={X.label_  for X in doc1.ents}
    ent2={X.label_  for X in doc2.ents}
    if len(ent1.union(ent2))>0:
        return len(ent1.intersection(ent2))/len(ent1.union(ent2))
    return 0.5

#classes to return text and numeric values for transformer

from sklearn.base import BaseEstimator, TransformerMixin
class TextSelector(BaseEstimator, TransformerMixin):
    def __init__(self, field):
        self.field = field
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.field]
class NumberSelector(BaseEstimator, TransformerMixin):
    def __init__(self, field):
        self.field = field
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[[self.field]]
    
    
def Tokenizer(str_input):
    words = re.sub(r"[^A-Za-z0-9\-]", " ", str_input).lower().split()
    return lemmatize(words)

def build_features(train):
    
    Counts_for_tf = defaultdict(int)
    
    for sent in train['Sentence1'].tolist():
        for w in [w.strip("?.,") for w in sent.split()]: Counts_for_tf[w] += 1
    for sent in train['Sentence2'].tolist():
        for w in [w.strip("?.,") for w in sent.split()]: Counts_for_tf[w] += 1
        
        
    

    #tokenization    
    train['tokenized_sent1'] = train.apply(lambda row: nltk.word_tokenize(row['Sentence1']), axis=1)
    train['tokenized_sent2'] = train.apply(lambda row: nltk.word_tokenize(row['Sentence2']), axis=1)
    
    #lemmatization
    train['lemmatized_sent1'] = train.apply(lambda row: lemmatize(row['tokenized_sent1']), axis=1)
    train['lemmatized_sent2'] = train.apply(lambda row: lemmatize(row['tokenized_sent2']), axis=1)
    
    #stop words removal    
    train['lemmatized_sent1'].apply(lambda x: [item for item in x if item not in stop])
    train['lemmatized_sent2'].apply(lambda x: [item for item in x if item not in stop])

    #POS tags    
    train['POS_Tags1'] = train.apply(lambda row: nltk.pos_tag(row['lemmatized_sent1']), axis=1)
    train['POS_Tags2'] = train.apply(lambda row: nltk.pos_tag(row['lemmatized_sent2']), axis=1)
    
    train['Root1'] = train.apply(lambda row: dependency_parse(row['Sentence1']), axis=1)
    train['Root2'] = train.apply(lambda row: dependency_parse(row['Sentence2']), axis=1)
    
    train['flagCheckRoot'] = train.apply(lambda row: row['Root1']==row['Root2'], axis=1)
    
    train['baseline_sim']=train.apply(lambda row: sentence_similarity_simple_baseline(row['Sentence1'],row['Sentence2'],Counts_for_tf), axis=1)
    train['word_align_sim']=train.apply(lambda row: sentence_similarity_word_alignment(row['Sentence1'],row['Sentence2']), axis=1)
    
    train['baseline_sim']=train.apply(lambda row: sentence_similarity_simple_baseline(row['Sentence1'],row['Sentence2'],Counts_for_tf), axis=1)
    train['word_align_sim']=train.apply(lambda row: sentence_similarity_word_alignment(row['Sentence1'],row['Sentence2']), axis=1)
    train['info_con_sim']=train.apply(lambda row: sentence_similarity_information_content(row['Sentence1'],row['Sentence2']), axis=1)
    train['tag_overlap']=train.apply(lambda row: extract_overlap_pen(row['Sentence1'],row['Sentence2']), axis=1)
    
    train['verb_diff']=train.apply(lambda row: verb_diff(row['Sentence1'],row['Sentence2']), axis=1)
    train['noun_diff']=train.apply(lambda row: noun_diff(row['Sentence1'],row['Sentence2']), axis=1)
    train['adj_diff']=train.apply(lambda row: adj_diff(row['Sentence1'],row['Sentence2']), axis=1)
    train['adv_diff']=train.apply(lambda row: adv_diff(row['Sentence1'],row['Sentence2']), axis=1)
    
    
    train['mmr']=train.apply(lambda row: extract_mmr_t(row['Sentence1'],row['Sentence2']), axis=1)
    train['res_sim']=train.apply(lambda row: extract_res_vec_similarity(row['Sentence1'],row['Sentence2']), axis=1)
    
    train['no_ques_diff']=train.apply(lambda row: (row['Sentence1'].count("?")-row['Sentence2'].count("?")), axis=1)
    train['no_excl_diff']=train.apply(lambda row: (row['Sentence1'].count("!")-row['Sentence2'].count("!")), axis=1)
    
    
    train['annotations_s1'] = train.apply(lambda row: get_annotation(row['Sentence1']), axis=1)
    train['annotations_s2'] = train.apply(lambda row: get_annotation(row['Sentence2']), axis=1)
    
    
    # NER features

    train['num_same_URI'] = train.apply(lambda row: duplicates(row['annotations_s1']['URI'],row['annotations_s2']['URI']), axis=1)
    # NER features

    train['overlap_types'] = train.apply(lambda row:get_common_types(row['annotations_s1']['types'],row['annotations_s2']['types']), axis=1)
    
    train['common_entities'] = train.apply(lambda row: get_common_entities(row['Sentence1'],row['Sentence2']), axis=1)




    
    

if __name__ == "__main__":
    #train_file=sys.argv[0]
    train_file='data/train-set.txt'
    train = pd.read_csv(train_file, sep="\t",error_bad_lines=False)
    train = train[np.isfinite(train['Gold Tag'])]  #remove nan values
    
    processed_train=build_features(train)
    
    X=processed_train[[ 'baseline_sim','Sentence1','Sentence2',
       'word_align_sim', 'info_con_sim', 'tag_overlap', 'res_sim', 'no_ques_diff', 'no_excl_diff', 'num_same_URI', 'overlap_types','common_entities', 'verb_diff', 'noun_diff', 'adj_diff', 'adv_diff']]
    Y = processed_train.loc[:, 'Gold Tag']
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)
    
    all_corpus=train['Sentence1'].tolist()+train['Sentence2'].tolist()
    
    classifier = Pipeline([
    ('features', FeatureUnion([
        ('tfidf1', Pipeline([
            ('colext', TextSelector('Sentence1')),
            ('tfidf', TfidfVectorizer(tokenizer=Tokenizer, stop_words=stop_words,
                     min_df=.0025, max_df=0.25, ngram_range=(1,2))),
            ('svd', TruncatedSVD(algorithm='randomized', n_components=300)), 
        ])),
       ('tfidf2', Pipeline([
            ('colext', TextSelector('Sentence2')),
            ('tfidf', TfidfVectorizer(tokenizer=Tokenizer, stop_words=stop_words,
                     min_df=.0025, max_df=0.25, ngram_range=(1,2))),
            ('svd', TruncatedSVD(algorithm='randomized', n_components=300)), 
        ])),
        ('no_ques_diff', Pipeline([
            ('wordext', NumberSelector('no_ques_diff')),
            ('wscaler', StandardScaler()),
        ])),
         ('no_excl_diff', Pipeline([
            ('wordext', NumberSelector('no_excl_diff')),
            ('wscaler', StandardScaler()),
        ])),
         ('baseline_sim', Pipeline([
            ('wordext', NumberSelector('baseline_sim')),
            ('wscaler', StandardScaler()),
        ])),
         ('word_align_sim', Pipeline([
            ('wordext', NumberSelector('word_align_sim')),
            ('wscaler', StandardScaler()),
        ])),
         ('res_sim', Pipeline([
            ('wordext', NumberSelector('res_sim')),
            ('wscaler', StandardScaler()),
        ])),
         ('num_same_URI', Pipeline([
            ('wordext', NumberSelector('num_same_URI')),
            ('wscaler', StandardScaler()),
        ])),
         ('info_con_sim', Pipeline([
            ('wordext', NumberSelector('info_con_sim')),
        
        ])),
           ('common_entities', Pipeline([
            ('wordext', NumberSelector('common_entities')),
        
        ])),
           ('verb_diff', Pipeline([
            ('wordext', NumberSelector('verb_diff')),
        
        ])),
         ('noun_diff', Pipeline([
            ('wordext', NumberSelector('noun_diff')),
        
        ])),
        ('adj_diff', Pipeline([
            ('wordext', NumberSelector('adj_diff')),
        
        ])),
        ('adv_diff', Pipeline([
            ('wordext', NumberSelector('adv_diff')),
        
        ])),
         ('overlap_types', Pipeline([
            ('wordext', NumberSelector('overlap_types')),
           
        ])),
           ('tag_overlap', Pipeline([
            ('wordext', NumberSelector('tag_overlap')),
           
        ]))
    ])),
#    ('clf', XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.1)),
   ('rf', RandomForestClassifier()),
  #('svr',SVR(kernel='linear')),
     ])
            

    
    classifier.fit(X_train, y_train)           
    preds = classifier.predict(X_test)
    
    return y_test,preds
            
            

