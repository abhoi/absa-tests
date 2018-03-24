import pandas as pd
import csv
from nltk import word_tokenize

# train data path
DATA1_TRAIN_PATH = 'data/data_1_train.csv'
DATA2_TRAIN_PATH = 'data/data_2_train.csv'

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

def preprocessing(tech_reviews, food_reviews):
	# import nltk
	# nltk.download('punkt') # download punkt if not available
	tech_reviews['tokenized_text'] = tech_reviews['text'].apply(word_tokenize)
	food_reviews['tokenized_text'] = food_reviews['text'].apply(word_tokenize)
	# remove stop words
	# remove punctuations...?
	# word2vec/glove (both)
	
	print(food_reviews['tokenized_text'][2302])

if __name__ == '__main__':
	tech_reviews, food_reviews = load_and_clean()
	preprocessing(tech_reviews, food_reviews)