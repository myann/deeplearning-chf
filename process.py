from data import Dataset
from models import *

import csv
import numpy as np
import pickle as pk

## SET THESE FIRST
INPUT_PATH = 'dataset/pmh_gensim.txt'
TRAIN_CSV = 'dataset/train_ids_dat.csv'
VALID_CSV = 'dataset/val_ids_dat.csv'
TEST_CSV = 'dataset/test_ids_dat.csv'

def loader():
    '''Loads the dataset from the appropriate files.'''
    train_rows = list(csv.DictReader(open(TRAIN_CSV)))
    valid_rows = list(csv.DictReader(open(VALID_CSV)))
    test_rows = list(csv.DictReader(open(TEST_CSV)))
    train_idx = np.array([row['d_ICES_patient_id'] for row in train_rows],
                         dtype=int)
    valid_idx = np.array([row['d_ICES_patient_id'] for row in valid_rows],
                         dtype=int)
    test_idx = np.array([row['d_ICES_patient_id'] for row in test_rows],
                        dtype=int)
    data = np.array([emr.rstrip('\n').split() for emr in open(INPUT_PATH)])
    labels = np.empty_like(data, dtype=float)
    labels[train_idx] = [row['OA'] for row in train_rows]
    labels[valid_idx] = [row['OA'] for row in valid_rows]
    labels[test_idx] = [row['OA'] for row in test_rows]
    return {
        'data': data,
        'labels': labels,
        'train_idx': train_idx,
        'valid_idx': valid_idx,
        'test_idx': test_idx
    }

if __name__ == '__main__':
    # Load the dataset, build dictionary and BoW representation
    dataset = Dataset(loader)
    dictionary = Dictionary(dataset.all)  #  vocab uses entire set

    # Doc2Vec model
    tagged = dataset.transform(DocTagger(), force_obj=True)
    d2v_dm = Doc2VecModel(tagged.all, min_count=1, window=10,
                       size=300, sample=1e-5)
    d2v_dm.fit(tagged.train)
    w2v_dm_transformer = Word2VecTransformer(d2v_dm)
    d2v_dbow = Doc2VecModel(tagged.all, min_count=1, window=10,
                       size=300, sample=1e-5, dm=0)
    d2v_dbow.fit(tagged.train)
    w2v_dbow_transformer = Word2VecTransformer(d2v_dm)

    corpus = dictionary.to_bow(dataset.train)  # only using training set

    # Fit TF-IDF, LDA and HDP models (these wrappers around Gensim models)
    idf = TFIDFModel(dictionary=dictionary.inst)  # corpus not needed
    lda = LDAModel(corpus, id2word=dictionary.inst, num_topics=100)
    hdp = HDPModel(corpus, id2word=dictionary.inst)

    corpus = dictionary.to_bow(dataset.all)  # now using all data

    # Store vector representations
    pk.dump(dataset.replace(corpus, transformer=idf), open('idf.pk', 'wb'))
    pk.dump(dataset.replace(corpus, transformer=lda), open('lda.pk', 'wb'))
    pk.dump(dataset.replace(corpus, transformer=hdp), open('hdp.pk', 'wb'))
    pk.dump(dataset.transform(d2v_dm), open('d2v_dm.pk', 'wb'))
    pk.dump(dataset.transform(d2v_dbow), open('d2v_dbow.pk', 'wb'))
    # Store mapped integer sequences - range [1, max_id]
    pk.dump(dataset.transform(dictionary), open('iseq.pk', 'wb'))
    # Store mapped vector sequences
    pk.dump(dataset.transform(w2v_dm_transformer), open('vseq_dm.pk', 'wb'))
    pk.dump(dataset.transform(w2v_dbow_transformer), open('vseq_dbow.pk', 'wb'))

    dictionary.inst.save('dict.pk')  # for later retrieval
