import numpy as np
import pandas as pd
import csv
import os
import pickle

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding

# train data path
DATA1_TRAIN_PATH = 'data/data_1_train.csv'
DATA2_TRAIN_PATH = 'data/data_2_train.csv'

# GLoVe pre-trained word vectors path
EMBEDDING_DIR = 'embeddings/'
EMBEDDING_TYPE = 'glove.6B.300d.txt' # glove.6B.300d.txt
EMBEDDING_PICKLE_DIR = 'embeddings_index.p'
EMBEDDING_ERROR_DIR = 'embeddings_error.p'

# tokenizer path
TOKENIZER_DIR = 'embeddings/tokenizer.p'

MAX_SEQ_LENGTH = 50
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 300
ASPECT_INDEX = 1

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

	print('tech_reviews shape: ' + str(tech_reviews.shape))
	print('food_reviews shape: ' + str(food_reviews.shape))
	return tech_reviews, food_reviews

def read_GLoVe_embeddings():
	embeddings_index = {}
	error_words = []

	# check if GLoVe pickle dump exists
	if not os.path.exists(EMBEDDING_DIR + EMBEDDING_PICKLE_DIR):
		# check if GLoVe embeddings exist
		if not os.path.exists(EMBEDDING_DIR):
			print('GloVe embedding does not exist...exiting.')
			exit(1)
		else:
			print(EMBEDDING_TYPE + ' embeddings found!')
			e = open(EMBEDDING_DIR+EMBEDDING_TYPE)

		# read GLoVe embedding matrix
		for line in e:
			values = line.split()
			word = values[0]
			try:
				coefs = np.asarray(values[1:], dtype='float32')
				embeddings_index[word] = coefs
			except Exception:
				error_words.append(word) # append error word
		e.close()

		# check for words that could not be loaded properly
		if len(error_words) > 0:
			print('%d words could not be added.' % (len(error_words)))
			print('Words are: \n', error_words)

		pickle.dump(embeddings_index, open(EMBEDDING_DIR + EMBEDDING_PICKLE_DIR, 'wb')) # dump embedding matrix
		pickle.dump(error_words, open(EMBEDDING_DIR + EMBEDDING_ERROR_DIR, 'wb')) # dump error words
	else:
		print(EMBEDDING_TYPE + ' pickle dump found!')
		embeddings_index = pickle.load(open(EMBEDDING_DIR + EMBEDDING_PICKLE_DIR, 'r')) # load embedding matrix
		error_words = pickle.load(open(EMBEDDING_DIR + EMBEDDING_ERROR_DIR, 'r')) # load error words

	return embeddings_index, error_words

def fit_tokenizer(tech_reviews):
	if not os.path.exists(TOKENIZER_DIR):
		# Tokenizer to fit on training text
		t = Tokenizer()
		t.fit_on_texts(tech_reviews['text'])
		pickle.dump(t, open(TOKENIZER_DIR, 'wb')) # dump the Tokenizer
	else:
		# Load pre-fit Tokenizer
		t = pickle.load(open(TOKENIZER_DIR, 'r'))

	VOCAB_SIZE = len(t.word_index) + 1
	word_index = t.word_index
	print('Found %s unique tokens' % len(word_index))
	return t, word_index, VOCAB_SIZE

def sub_list_finder(sentence, aspect):
	global ASPECT_INDEX
	sentence_length = len(sentence)
	aspect_sequence = np.zeros(shape=(sentence_length))
	for i in range(sentence_length):
		if aspect[0] == sentence[i]:
			sub_list = sentence[i:i+len(aspect)]
			if aspect == sub_list:
				for j in range(len(aspect)):
					aspect_sequence[i+j] = ASPECT_INDEX
	ASPECT_INDEX += 1
	return aspect_sequence

def get_aspect_sequences(word_sequence, aspects):
	aspect_sequence = np.array(map(lambda x, y: sub_list_finder(x, y), word_sequence, aspects)) # create aspect sequences based on word_sequence and aspects
	return aspect_sequence

def load_embedding_matrix(dataset):
	embeddings_index, error_words = read_GLoVe_embeddings()
	t, word_index, VOCAB_SIZE = fit_tokenizer(dataset)

	aspects = np.array(dataset['aspect_term'])
	text = np.array(dataset['text'])
	sequences = t.texts_to_sequences(text) # text to sequence (encoded text)
	word_sequence = np.array(map(lambda x: text_to_word_sequence(x), text)) # text to word sequence
	aspects = np.array(map(lambda x: text_to_word_sequence(x), aspects)) # aspects to word sequence
	padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQ_LENGTH, dtype='int32', padding='post') # sequence to padded sequence
	aspect_sequences = get_aspect_sequences(word_sequence, aspects)

	print('Creating weight matrices...')
	# create weight matrix for words in text
	embedding_matrix = np.zeros((MAX_NB_WORDS, EMBEDDING_DIM))
	for word, i in t.word_index.items():
		embedding_vector = embeddings_index.get(word)
		if embedding_vector is not None:
			# words not found in embedding index will be all-zeros
			embedding_matrix[i] = embedding_vector

	print(embedding_matrix.shape)

	# load pre-trained word embeddings into an Embedding layer
	embedding_layer = Embedding(MAX_NB_WORDS, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQ_LENGTH, trainable=False)
	embedding_layer_aspects = Embedding(MAX_NB_WORDS, 50, input_length=50, trainable=True, mask_zero=True)

	print('Embedding layer set...')

if __name__ == '__main__':
	tech_reviews, food_reviews = load_and_clean()
	load_embedding_matrix(tech_reviews)

	# aspect term, train weights, unique IDs, concat layer
	# mark-zero