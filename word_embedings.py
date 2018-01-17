# -*- coding: utf-8 -*-

import re
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import numpy as np

TRAINING_SENTENCES = 200
TRAINING_FILE ='embeddings_training/training_for_embeddings_'+str(TRAINING_SENTENCES) 

stop = stopwords.words('spanish')

def clean_dataset_for_embedings(path,output):
    i = open(path,'r')
    o = open(output,'w')
    for w in range(TRAINING_SENTENCES):
        if w % 100 == 0:
            print w*1.0/TRAINING_SENTENCES*100,'%'
        s = i.readline()
        if len(s) > 6 :
            #print s
            s = clean_str(s)
            #print s
            o.write(s)
    i.close()
    o.close()

def load_word2vec(path,binary = False):
    from gensim.models.keyedvectors import KeyedVectors
    word_vectors = KeyedVectors.load_word2vec_format(path, binary=binary)
    #print word_vectors['nada']
    return word_vectors

def clean_str(str_input):
    str_input = str_input.lower()
    text = [w for w in str_input.split() if w not in stop]
    str_input = ''
    for w in text:
        str_input = str_input + ' ' + w

    str_accent =  ['año' ,'á','é','è','í','ó','ú','ñ','%','#','@','"',"'",'/','-','°','(',')','[',']','.',',',':',';','ç','Ò','²','«','»','!','¡','¿','?','*','^']
    str_replace = ['anio','a','e','e','i','o','u','n','' ,'' ,'' ,'' ,'' ,'' ,'' ,'' ,'' ,'' ,'' ,'' ,'' ,'' ,'' ,'' ,'c','o','2','' ,'' ,'' ,'' ,'' ,'' ,'' ,'' ]
    #str_input = str_input.encode("utf-8")
    for s in range(len(str_accent)):
        str_input = str_input.replace(str_accent[s],str_replace[s])
    
    return re.sub(' +',' ',str_input).rstrip()

def train_word2vec(path):
    i = open(path,'r')
    data = []
    for l in range(TRAINING_SENTENCES):
        s = i.readline()
        data.append(s.split(' '))
    i.close()
    model = Word2Vec(data, size=300, window=5, min_count=1, workers=4)
    model.wv.save_word2vec_format('embeddings_models/model_word2vec_'+str(TRAINING_SENTENCES),binary = False)
    return model

def train_fasttext(path):
    # https://pypi.python.org/pypi/fasttext
    import fasttext
    model = fasttext.skipgram(path, 'embeddings_models/model_fasttext_'+str(TRAINING_SENTENCES),dim=300)
    
def train_glove(path):
    import itertools
    from gensim.models.word2vec import Text8Corpus
    from gensim.scripts.glove2word2vec import glove2word2vec
    from glove import Corpus, Glove
    #import os
    #import struct
    sentences = list(itertools.islice(Text8Corpus(path),None))
    corpus = Corpus()
    corpus.fit(sentences, window=10)
    glove = Glove(no_components=300, learning_rate=0.05)
    glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
    glove.add_dictionary(corpus.dictionary)
    file_name = 'embeddings_models/model_glove_'+str(TRAINING_SENTENCES)
    glove.save(file_name)
    glove2word2vec(file_name, file_name +'_modified')
    """
    command = 'python -m gensim.scripts.glove2word2vec -i ' +file_name+' -o '+file_name+'_modified'
    os.system(command)
    with open(file_name+'_modified', mode='rb') as file: # b is important -> binary
        fileContent = file.read()
        print 'Content',fileContent
    """
    print 'Finished'
    return glove

def get_word2vec_encode(word):
    if word in m_word2vec:
        return m_word2vec[word]
    return None

def get_glove_encode(word):
    if word in m_glove:
        return m_glove[word]

def get_fasttext_encode(word):
    if word in m_fasttext:
        return m_fasttext[word]

def build_data_set_from_xml(xml_path):
    SENTIMENT_POSITIVE = 'P'
    SENTIMENT_NEGATIVE = 'N'
    SENTIMENT_NONE = 'NONE'
    SENTIMENT_NEUTRAL = 'NEU'

    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_path)
    root = tree.getroot()
    data = []
    for tweet in root.findall('tweet'):
        temp = []
        content = tweet.find('content').text
        content = content.encode("utf-8")
        content = clean_str(content)
        temp.append(content)
        polarity = tweet.find('sentiment').find('polarity').find('value').text
        if polarity == SENTIMENT_POSITIVE:
            temp.append(1)
        elif polarity == SENTIMENT_NEGATIVE:
            temp.append(-1)
        elif polarity == SENTIMENT_NEUTRAL:
            temp.append(0)
        else :
            temp.append(2)
        data.append(temp)
    return data

def build_data_set_test_from_xml(xml_path,qrel_path):
    qrel = file(qrel_path,'r')
    SENTIMENT_POSITIVE = 'P'
    SENTIMENT_NEGATIVE = 'N'
    SENTIMENT_NONE = 'NONE'
    SENTIMENT_NEUTRAL = 'NEU'

    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_path)
    root = tree.getroot()
    data = []
    for tweet in root.findall('tweet'):
        temp = []
        content = tweet.find('content').text
        content = content.encode("utf-8")
        content = clean_str(content)
        temp.append(content)
        #polarity = tweet.find('sentiment').find('polarity').find('value').text
        polarity = qrel.readline().split()[1]
        if polarity == SENTIMENT_POSITIVE:
            temp.append(1)
        elif polarity == SENTIMENT_NEGATIVE:
            temp.append(-1)
        elif polarity == SENTIMENT_NEUTRAL:
            temp.append(0)
        else :
            temp.append(2)
        data.append(temp)
    qrel.close()
    return data
    
#%% Get batch for training
# Call Example 
# batch,sentiment = getBatch(data,1,10)
def get_batch(data,start,end):
    max_size = 40
    batch = []
    sentiment = []
    for b in range(start,end):
        dim_1 = []
        dim_2 = []
        dim_3 = []
        content,s = data[b]
        words = content.split(' ')
        sentence = []
        for w in words:
            d_1 = get_word2vec_encode(w)
            d_2 = get_glove_encode(w)
            d_3 = get_fasttext_encode(w)
            embed = []
            if d_1 is not None and d_2 is not None and d_3 is not None:
                embed.append(d_1)
                embed.append(d_2)
                embed.append(d_3)
                sentence.append(np.array(embed).transpose())
        if sentence:
            if len(sentence) < max_size:
                deff = max_size - len(sentence)
                embed = []
                embed.append([0.0]*300)
                embed.append([0.0]*300)
                embed.append([0.0]*300)
                for i in range(deff):
                    sentence.append(np.array(embed).transpose())
            batch.append(sentence)
            sentiment.append(s)
    return np.array(batch),np.array(sentiment)

def load_fast_text(path):
    import fasttext
    model = fasttext.load_model(path, encoding='utf-8')
    return model

def load_glove(path):
    #python -m gensim.scripts.glove2word2vec -i model_glove_1000 -o model_glove_1000_modified
    #return Word2Vec.load(path)
    #return load_word2vec(path,binary = True)
    from gensim.models.keyedvectors import KeyedVectors
    word_vectors = KeyedVectors.load_word2vec_format(path, binary=True)
    return word_vectors

def train_embeddings():
    clean_dataset_for_embedings('/home/alonzo/Documentos/Projects/wikipedia_dataset.txt',TRAINING_FILE)
    print 'Start fasttext'
    train_fasttext(TRAINING_FILE)
    #print 'Start glove'
    #train_glove(TRAINING_FILE)
    print 'Start word2vec'
    train_word2vec(TRAINING_FILE)

#train_embeddings()

m_fasttext = load_word2vec('embeddings_models/wiki.es_ligth.vec',binary = False)
m_word2vec = load_word2vec('embeddings_models/SBW-vectors-300-min5_ligth',binary = False)
m_glove = load_word2vec('embeddings_models/glove_combine_ligth',binary = False)

def combine_models(model_1,model_2):
    new_model = {}
    keys = []
    for w in model_1.vocab:
        keys.append(w)
    for w in keys:
        if w in model_1 and w in model_2:
            new_model[w] = (model_1[w] + model_2[w] )/ 2
    return new_model

vocabulary = {}
def add_to_vocabulary(data,model):
    for d,s in data:
        words = d.split()
        for w in words:
            if w not in vocabulary and w in model:
                vocabulary[w] = model[w]

def save_new_model_from_vocabulary(path):
    model = file(path,'w')
    keys = vocabulary.keys()
    model.write(str(len(keys))+' 300\n')
    for w in vocabulary:
        model.write(w)
        w_v = vocabulary[w]
        for e in w_v:
            model.write(' '+str(e))
        model.write('\n')
    model.close()

#m_glove = combine_models(m_fasttext,m_word2vec)
#add_to_vocabulary(build_data_set_from_xml('datasets/intertass-train-tagged.xml'),m_glove)
#add_to_vocabulary(build_data_set_from_xml('datasets/intertass-development-tagged.xml'),m_glove)
#add_to_vocabulary(build_data_set_from_xml('datasets/intertass-test.xml'),m_glove)
#save_new_model_from_vocabulary('embeddings_models/glove_combine_ligth')

#data = build_data_set_test_from_xml('datasets/intertass-test.xml','datasets/intertass-sentiment.qrel')
#batch,sentiment = get_batch(data,0,1898)
#print sentiment.shape
#print batch.shape