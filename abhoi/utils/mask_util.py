import numpy as np
import pandas as pd
import csv
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split

# train data path
DATA1_TRAIN_PATH = '../data/data_1_train.csv'
DATA2_TRAIN_PATH = '../data/data_2_train.csv'

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

def get_unique_id_for_aspects(data):
    aspect_dict = {}
    i = 0
    for i in range(0,len(data)):
        aspect_term = data['aspect_term'][i]
        if aspect_term not in aspect_dict.keys():
            aspect_dict[aspect_term] = i
            i = i + 1
    return aspect_dict


def get_aspect_encoded_list(sentence,aspect,index):
    
    words = nltk.word_tokenize(sentence)
    aspects = nltk.word_tokenize(aspect)
    encoded_sentence = []
    num_indecies_skip = 0
    for i in range(0,len(words)):
        if aspects[0] == words[i]:
            aspect_in_sent = words[i:i+len(aspects)]
            if aspect_in_sent == aspects:
                num_indecies_skip = len(aspects)-1
                for j in range(0,len(aspects)):
                    # encoded_sentence.append(index)
                    encoded_sentence.append(1)
                    
                continue
            else:
                if(num_indecies_skip != 0):
                    num_indecies_skip = num_indecies_skip - 1
                    continue
                encoded_sentence.append(0)
        else:
            if(num_indecies_skip != 0):
                    num_indecies_skip = num_indecies_skip - 1
                    continue
            encoded_sentence.append(0)
    
    pad_length = 120 - len(encoded_sentence)

    return np.concatenate((np.array(encoded_sentence),np.zeros(pad_length)))




tech_reviews, food_reviews = load_and_clean()


aspect_dict = get_unique_id_for_aspects(tech_reviews)
food_aspect_dict = get_unique_id_for_aspects(food_reviews)

encoded_sentences = []
for i in range(0,len(tech_reviews)):
    sentence = tech_reviews['text'][i]
    aspect_term = tech_reviews['aspect_term'][i]
    aspect_encoded_sentence = get_aspect_encoded_list(sentence,aspect_term,aspect_dict[aspect_term])
    # aspect_encoded_sentence = get_aspect_encoded_list(sentence,aspect_term,i)
    
    encoded_sentences.append(aspect_encoded_sentence)
tech_reviews['encoded_text'] = encoded_sentences

encoded_sentences = []
for i in range(0,len(food_reviews)):
    sentence = food_reviews['text'][i]
    aspect_term = food_reviews['aspect_term'][i]
    aspect_encoded_sentence = get_aspect_encoded_list(sentence,aspect_term,food_aspect_dict[aspect_term])
    # aspect_encoded_sentence = get_aspect_encoded_list(sentence,aspect_term,i)
    
    encoded_sentences.append(aspect_encoded_sentence)
food_reviews['encoded_text'] = encoded_sentences

# print(tech_reviews[])



# msk = np.random.rand(len(tech_reviews)) < 0.8
# train = tech_reviews[msk]
# test = tech_reviews[~msk]

count_vect = CountVectorizer()
# X_train_counts = count_vect.fit_transform(train['text'])

# tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
# X_train_tf = tf_transformer.transform(X_train_counts)
# X_train_tf.shape

vectorizer = TfidfVectorizer(max_df=1.0, min_df=1, stop_words='english',norm = None)
X = vectorizer.fit_transform(tech_reviews['text'])
food_X = vectorizer.fit_transform(food_reviews['text'])
# X_t = vectorizer.fit_transform(test['text'])

tfidf_transformer = TfidfTransformer()
# X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)


print(X.toarray().shape)
X_train = X.toarray()

train_lists = []
food_train_lists = []

for i in range(0,X.toarray().shape[0]):
    train_lists.append(np.concatenate((X_train[i],tech_reviews['encoded_text'][i])))
    
for i in range(0,food_X.toarray().shape[0]):
    food_train_lists.append(np.concatenate((food_X.toarray()[i],food_reviews['encoded_text'][i])))

print('Concatenation done')
tech_reviews['feature'] = train_lists
food_reviews['feature'] = food_train_lists
# print(tech_reviews['feature'])



X_train, X_test, y_train, y_test = train_test_split(train_lists, tech_reviews['class'].astype(int), test_size=0.2, random_state=42)
food_X_train, food_X_test, food_y_train, food_y_test = train_test_split(food_train_lists, food_reviews['class'].astype(int), test_size=0.2, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(X.toarray(), tech_reviews['class'], test_size=0.2, random_state=42)


# X_feature = vectorizer.fit_transform(tech_reviews['feature'])

# clf = GaussianNB().fit(X_train, y_train.astype(int))

from sklearn import svm
from sklearn.cross_validation import cross_val_score, cross_val_predict, KFold
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


model = svm.SVC(kernel='linear', C = 1.0)




from sklearn.tree import DecisionTreeClassifier
clft = DecisionTreeClassifier(random_state=0)

# clfr = RandomForestClassifier(max_depth=2, random_state=0)
print('Training SVM')


# clf = GaussianNB().fit(np.array(X_train), y_train.astype(int))
clf = GaussianNB()

# scores = cross_val_score(model, X_train, y_train, cv=10)
scores = cross_val_score(clft, food_X_train, food_y_train, cv=10)
print('SVM training done')
print(scores)




# X_new_counts = count_vect.transform(test['text'])
# X_new_tfidf = tfidf_transformer.transform(X_new_counts)

# predicted = clf.predict(X_train)
# print(np.mean(predicted == y_train.astype(int)))

# model.fit(np.array(X_train), y_train.astype(int))
# model.score(np.array(X_train), y_train.astype(int))

# predicted= model.predict(np.array(X_test))
print('Going to predict...')
# train_predictions = cross_val_predict(model, X_train, y_train, cv=10)
# test_predictions = cross_val_predict(model, X_test, y_test, cv=10)
# print('Train Accuracy: ',np.mean(train_predictions == y_train))
# print('Test Accuracy: ',np.mean(test_predictions == y_test))


train_predictions = cross_val_predict(clft, food_X_train, food_y_train, cv=10)
test_predictions = cross_val_predict(clft, food_X_test, food_y_test, cv=10)
print('Train Accuracy: ',np.mean(train_predictions == food_y_train))
print('Test Accuracy: ',np.mean(test_predictions == food_y_test))



def get_metrics(predictions,labels,class_type):

    TP = 0.0
    FP = 0.0
    FN = 0.0
    i = 0
    for label in labels:
        if predictions[i] == class_type and label == class_type:
            TP = TP + 1
        if predictions[i] == class_type and label != class_type:
            FP = FP + 1
        if predictions[i] != class_type and label == class_type:
            FN = FN + 1
        i = i + 1

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    fscore = (2 * precision * recall)/(precision + recall)
    print('Precision: ',precision)
    print('Recall: ',recall)
    print('fscore: ',fscore)
    print('**********************')
    return precision,recall,fscore

# print('Training Metrics')
# print('Postitive Class')
# get_metrics(train_predictions,y_train,1)
# print('Negative Class')
# get_metrics(train_predictions,y_train,-1)

# print('Test Metrics')
# print('Postitive Class')
# get_metrics(test_predictions,y_test,1)
# print('Negative Class')
# get_metrics(test_predictions,y_test,-1)

print('------------------------------------')

print('Training Metrics')
print('Postitive Class')
get_metrics(train_predictions,food_y_train,1)
print('Negative Class')
get_metrics(train_predictions,food_y_train,-1)
print('Neutral Class')
get_metrics(train_predictions,food_y_train,0)

print('Test Metrics')
print('Postitive Class')
get_metrics(test_predictions,food_y_test,1)
print('Negative Class')
get_metrics(test_predictions,food_y_test,-1)
print('Neutral Class')
get_metrics(test_predictions,food_y_test,0)