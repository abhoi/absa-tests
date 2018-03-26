import numpy as np
import pandas as pd
import csv
import os
import sys
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding


# train data path
DATA1_TRAIN_PATH = 'data/data_1_train.csv'
DATA2_TRAIN_PATH = 'data/data_2_train.csv'

# GLoVe pre-trained word vectors
EMBEDDING_DIR = 'embeddings/'
EMBEDDING_TYPE = 'glove.840B.300d.txt' # glove.6B.300d.txt

MAX_NB_WORDS = 95000

def load_and_clean():
	# read into pandas csv
	tech_reviews = pd.read_csv(DATA1_TRAIN_PATH, quoting=csv.QUOTE_NONE, error_bad_lines=False, skipinitialspace=True)
	food_reviews = pd.read_csv(DATA2_TRAIN_PATH, quoting=csv.QUOTE_NONE, error_bad_lines=False)
	
	# rename columns to remove whitespaces
	tech_reviews.columns = ['example_id', 'text', 'aspect_term', 'term_location', 'class']
	food_reviews.columns = ['example_id', 'text', 'aspect_term', 'term_location', 'class']
	
	# replace _ with whitespace and [comma] with ,
	tech_reviews['text'] = tech_reviews['text'].str.replace('_ ', '')
	food_reviews['text'] = food_reviews['text'].str.replace('_ ', '')
	tech_reviews['text'] = tech_reviews['text'].str.replace("\[comma\]", ',')
	food_reviews['text'] = food_reviews['text'].str.replace("\[comma\]", ',')

	print(tech_reviews.shape)
	print(food_reviews.shape)
	return tech_reviews, food_reviews

def load_embedding_matrix(tech_reviews):
	embeddings_index = {}
	error_words = []

	# Tokenizer to fit on training text
	t = Tokenizer()
	t.fit_on_texts(tech_reviews['text'])
	VOCAB_SIZE = len(t.word_index) + 1
	# text to sequence conversion
	tech_reviews['encoded_text'] = tech_reviews['text'].apply(t.texts_to_sequences)
	# pad sequences
	tech_reviews['padded_text'] = tech_reviews['encoded_text'].apply(pad_sequences)
	
	print('Encoding and padding done...')
	
	# check if GLoVe embeddings exist
	if not os.path.exists(EMBEDDING_DIR):
		print('GloVe embedding does not exist')
		sys.exit(1)
	else:
		print('GLoVE embedding found')
		e = open(EMBEDDING_DIR+EMBEDDING_TYPE)

	# read GLoVe embedding matrix
	for line in e:
		values = line.split()
		word = values[0]
		try:
			coefs = np.asarray(values[1:], dtype='float32')
			embeddings_index[word] = coefs
		except Exception:
			error_words.append(word)
	e.close()

	# check for words that could not be loaded properly
	if len(error_words) > 0:
		print('%d words could not be added.' % (len(error_words)))
		print('Words are: \n', error_words)

	# create weight matrix for words in text
	embedding_matrix = zeros((MAX_NB_WORDS, 100))
	for word, i in t.word_index.items():
		embedding_vector = embeddings_index.get(word)
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector

	print(embedding_matrix.shape)

if __name__ == '__main__':
	tech_reviews, food_reviews = load_and_clean()
	load_embedding_matrix(tech_reviews)