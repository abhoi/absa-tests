from scipy.stats import itemfreq
from sklearn.model_selection import StratifiedKFold

from keras_utils.keras_utils import *

from keras.utils.np_utils import to_categorical
from keras.layers import Input, Embedding, Dense, GlobalAveragePooling1D, Flatten
from keras.layers import add, multiply, LSTM, Bidirectional, BatchNormalization, LeakyReLU, concatenate, Lambda
from keras.models import Model
from keras import backend as K


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


class MaskedGlobalAveragePooling1D(GlobalAveragePooling1D):

    def __init__(self, **kwargs):
        super(MaskedGlobalAveragePooling1D, self).__init__(**kwargs)
        self.supports_masking = True


class MaskableFlatten(Flatten):

    def __init__(self, **kwargs):
        super(MaskableFlatten, self).__init__(**kwargs)
        self.supports_masking = True


# train data path
DATA1_TRAIN_PATH = '../data/data_1_train.csv'
DATA2_TRAIN_PATH = '../data/data_2_train.csv'

# GLoVe pre-trained word vectors path
EMBEDDING_DIR = '../embeddings/'
EMBEDDING_TYPE = 'glove.6B.300d.txt'  # glove.6B.300d.txt
EMBEDDING_PICKLE_DIR = 'embeddings_index.p'
EMBEDDING_ERROR_DIR = 'embeddings_error.p'
ASPECT_EMBEDDING_DIR = 'aspect_embeddings.p'

# tokenizer path
TOKENIZER_DIR = 'embeddings/tokenizer.p'

MAX_SEQ_LENGTH = 50
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 300

# aspect dictionary
aspect_dict = {}

"""
What this model does:

2 ip - 1 op model : 2 ip = sentence and aspect sentence

Shared embedding layer = reduce # of params and chance to overfit.
sentence embedding = sentence passed through embedding layer (keep for later)
aspect embedding = aspect sentence passed through embedding layer 

On this aspect embedding, use attention mechanism to jointly learn what is the "best" augmentation to the sentence embedding
-   Dense layer that maps 1 : 1 between the aspect embedding and the aspect attention
    -   Softmax forces it to choose the "parts" of the sentence that help the most in training
    -   No bias needed for attention

-   Next is to actually augment the aspect embeddings with this learned attention
    -   The element-wise multiplication forces many embeddings to become close to zero
    -   Only a few will remain "strong" after this multiplication. These are the "important" words in the aspect sentence

Finally, augment the original sentence embeddings with the attended aspect embeddings
-   This will "add" some strength to the embeddings of the "important" words
-   Remaining words will not be impacted at all (since they are added with near zero values)

Benefits of this model
-   Choose if you want to send a unique aspect sentence for the corresponding sentence
    -   By this I mean, you have a choice
    -   1) Use the original sentence as aspect input.
            In doing so, it is basically like saying learn on your own what the aspect word is
            It may not give much benefit, as the attended vector has the chance of being all equal (no attention)
    -   2) Use a true aspect encoding as the aspect input.
            Since you are sharing the embedding now, you cannot use random / own assigned aspects anymore.
            The aspect ids that you pass will now be from the original embedding matrix using the word_index
            dict that Keras gives you.

            In this case, an aspect sentence would be of the form : 
            [0 0 ... 32506 66049 5968 0 0 ...] 
            Here 32506 = "Apple", 66049 = "Macbook" 5968 = "Pro" (say)

"""

NUM_CLASSES = 3  # 0 = neg, 1 = neutral, 2 = pos

MAX_SENTENCE_LENGTH = 50
MAX_NUM_WORDS = 20000  # this will be number of unique "words" (n-grams etc) there are
MAX_NUM_ASPECT_WORDS = 300  # this will be the number of unique aspect "words" (uni-grams only)

EMBEDDING_DIM = 300
EMBEDDING_WEIGHTS = None

MASK_ZEROS = True  # this can be true ONLY for RNN models. If even 1 CNN is there, it will crash

#
# embedding = Embedding(MAX_NUM_WORDS, output_dim=EMBEDDING_DIM, mask_zero=MASK_ZEROS,
#                       weights=EMBEDDING_WEIGHTS, trainable=False)
#
# sentence_ip = Input(shape=(MAX_SENTENCE_LENGTH,), dtype='int32')
# aspect_ip = Input(shape=(MAX_SENTENCE_LENGTH,), dtype='int32')
#
# sentence_embedding = embedding(sentence_ip)  # Note: these are same embedding layer
# aspect_embedding = embedding(aspect_ip)  # Note: these are same embedding layer
#
# # Create the attention vector for the aspect embeddings
# aspect_attention = Dense(EMBEDDING_DIM, activation='softmax', use_bias=False,
#                          name='aspect_attention')(aspect_embedding)
#
# # dampen the aspect embeddings according to the attention with an element-wise multiplication
# aspect_embedding = multiply([aspect_embedding, aspect_attention])
#
# # augment the sample embedding with information from the attended aspect embedding
# sentence_embedding = add([sentence_embedding, aspect_embedding])
#
# # now you can continue with whatever layer other than CNNs
#
# x = LSTM(100)(sentence_embedding)
# x = Dense(NUM_CLASSES, activation='softmax')(x)
#
# model = Model(inputs=[sentence_ip, aspect_ip], outputs=x)
#
# model.summary()
#
#
# from keras.utils.vis_utils import plot_model
# plot_model(model, to_file='shared_embedding.png', show_shapes=False, show_layer_names=True)
#

"""
What this model does:

2 ip - 1 op model : 2 ip = sentence and aspect sentence

Disjoing embedding layer = more # of params and chance to overfit.
sentence embedding = sentence passed through embedding layer (keep for later ; not learned)
aspect embedding = aspect sentence passed through embedding layer (learned)

Benefits of this model
-  Use a true aspect encoding as the aspect input.
   Since you are learning the embedding now, you can use own assigned aspects.
            
   In this case, an aspect sentence would be of the form : 
   [0 0 ... 2 2 2 0 0 ...] 
   Here 2 = "Apple", 2 = "Macbook" 2 = "Pro" (say)
   Therefore, the id is given by you, and is shared over all of the aspect words for a given aspect term.

"""


def output_shape(input_shape):
    shape = list(input_shape)
    shape[-1] /= 2
    print(shape)
    return tuple(shape)


def model_2():
    K.clear_session()
    tech_reviews, food_reviews = load_and_clean()
    embedding_matrix, aspect_sequences, padded_sequences, labels = load_embedding_matrix(food_reviews)
    # labels = [x+1 for x in labels]
    print(itemfreq(labels))

    indices = np.arange(0, padded_sequences.shape[0], step=1, dtype=int)
    np.random.shuffle(indices)
    padded_sequences = padded_sequences[indices]
    labels = to_categorical(labels, num_classes=NUM_CLASSES)
    labels = labels[indices]
    aspect_sequences = aspect_sequences[indices]

    sentence_embedding = Embedding(MAX_NUM_WORDS, output_dim=EMBEDDING_DIM, mask_zero=MASK_ZEROS,
                                   weights=EMBEDDING_WEIGHTS, trainable=False)

    # aspect_embedding = Embedding(MAX_NUM_ASPECT_WORDS, EMBEDDING_DIM, mask_zero=MASK_ZEROS, trainable=True)
    #  this needs to be True
    aspect_embedding = Embedding(len(aspect_dict) + 1, EMBEDDING_DIM, mask_zero=MASK_ZEROS, trainable=True)

    sentence_ip = Input(shape=(MAX_SENTENCE_LENGTH,), dtype='int32')
    aspect_ip = Input(shape=(MAX_SENTENCE_LENGTH,), dtype='int32')

    sentence_embedding = sentence_embedding(sentence_ip)  # Note: these are two different embeddings
    aspect_embedding = aspect_embedding(aspect_ip)  # Note: these are two different embeddings

    # Create the attention vector for the aspect embeddings
    aspect_attention = Dense(EMBEDDING_DIM, activation='sigmoid', use_bias=False,
                             name='aspect_attention')(aspect_embedding)

    # dampen the aspect embeddings according to the attention with an element-wise multiplication
    aspect_embedding = multiply([aspect_embedding, aspect_attention])
    # augment the sample embedding with information from the attended aspect embedding
    sentence_embedding = concatenate([sentence_embedding, aspect_embedding])

    # now you can continue with whatever layer other than CNNs

    # x = MaskedGlobalAveragePooling1D()(sentence_embedding)
    # x = MaskableFlatten()(sentence_embedding)
    x = LSTM(256)(sentence_embedding)
    # y = Lambda(lambda z: z[:, :, :NUM_CELLS//2], output_shape=output_shape)(x)
    # x = Dense(NUM_CELLS//2, activation='softmax', use_bias=False)(x)

    # x = multiply([x, y])
    # x = MaskedGlobalAveragePooling1D()(x)
    # x = Dense(256, activation='linear', kernel_initializer='he_normal')(x)
    # x = BatchNormalization()(x)
    # x = LeakyReLU()(x)
    x = Dense(3, activation='softmax')(x)
    model = Model(inputs=[sentence_ip, aspect_ip], outputs=x)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

    print(model.summary())

    model.fit([padded_sequences, aspect_sequences], labels, epochs=10, verbose=1, validation_split=0.2)

    # from keras.utils.vis_utils import plot_model
    # plot_model(model, to_file='learned_embedding.png', show_shapes=False, show_layer_names=True)


def model_2_CV():
    K.clear_session()
    tech_reviews, food_reviews = load_and_clean()
    embedding_matrix, aspect_sequences, padded_sequences, labels = load_embedding_matrix(tech_reviews)
    labels = np.array([x + 1 for x in labels])
    print(itemfreq(labels))

    # Random shuffling of padded, aspect sequences and labels
    # indices = np.arange(0, padded_sequences.shape[0], step=1, dtype=int)
    # np.random.shuffle(indices)
    # padded_sequences = padded_sequences[indices]
    # labels = to_categorical(labels, num_classes=NUM_CLASSES)
    # labels = labels[indices]
    # aspect_sequences = aspect_sequences[indices]
    print(labels.shape)

    N_FOLDS = 3
    fbeta_scores = []
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=1000)
    for j, (train_idx, test_idx) in enumerate(skf.split(padded_sequences, labels)):
        print('Fold %d' % (j + 1))
        sentence_train, aspect_train, y_train = padded_sequences[train_idx], aspect_sequences[train_idx], \
                                                labels[train_idx]
        sentence_test, aspect_test, y_test = padded_sequences[test_idx], aspect_sequences[test_idx], labels[test_idx]

        y_train = to_categorical(y_train, 3)
        y_test = to_categorical(y_test, 3)

        sentence_embedding = Embedding(MAX_NUM_WORDS, output_dim=EMBEDDING_DIM, mask_zero=MASK_ZEROS,
                                       weights=EMBEDDING_WEIGHTS, trainable=False)
        aspect_embedding = Embedding(len(aspect_dict) + 1, EMBEDDING_DIM, mask_zero=MASK_ZEROS, trainable=True)

        sentence_ip = Input(shape=(MAX_SENTENCE_LENGTH,), dtype='int32')
        aspect_ip = Input(shape=(MAX_SENTENCE_LENGTH,), dtype='int32')

        sentence_embedding = sentence_embedding(sentence_ip)  # Note: these are two different embeddings
        aspect_embedding = aspect_embedding(aspect_ip)  # Note: these are two different embeddings

        # Create the attention vector for the aspect embeddings
        aspect_attention = Dense(EMBEDDING_DIM, activation='sigmoid', use_bias=False,
                                 name='aspect_attention')(aspect_embedding)
        # dampen the aspect embeddings according to the attention with an element-wise multiplication
        aspect_embedding = multiply([aspect_embedding, aspect_attention])
        # augment the sample embedding with information from the attended aspect embedding
        sentence_embedding = concatenate([sentence_embedding, aspect_embedding])
        x = LSTM(256)(sentence_embedding)
        x = Dense(3, activation='softmax')(x)
        model = Model(inputs=[sentence_ip, aspect_ip], outputs=x)

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc', fbeta_score])

        print(model.summary())

        model.fit([sentence_train, aspect_train], y_train, epochs=5, verbose=1,
                  validation_data=([sentence_test, aspect_test], y_test))

        scores = model.evaluate([sentence_test, aspect_test], y_test)
        fbeta_scores.append(scores[-1])

    print("Average fbeta score : ", sum(fbeta_scores) / len(fbeta_scores))


def model_3():
    K.clear_session()
    tech_reviews, food_reviews = load_and_clean()
    embedding_matrix, aspect_sequences, padded_sequences, labels = load_embedding_matrix(food_reviews)
    labels = np.array([x + 1 for x in labels])
    print(itemfreq(labels))

    N_FOLDS = 10
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=1000)
    f = open('history.txt', 'w+')
    for j, (train_idx, test_idx) in enumerate(skf.split(padded_sequences, labels)):
        print('Fold %d' % (j + 1))
        sentence_train, y_train = padded_sequences[train_idx], labels[train_idx]
        sentence_test, y_test = padded_sequences[test_idx], labels[test_idx]

        y_train = to_categorical(y_train, 3)
        y_test = to_categorical(y_test, 3)

        sentence_embedding = Embedding(MAX_NUM_WORDS, output_dim=EMBEDDING_DIM, mask_zero=MASK_ZEROS,
                                       weights=EMBEDDING_WEIGHTS, trainable=False)
        # labels = to_categorical(labels, 3)
        sentence_ip = Input(shape=(MAX_SENTENCE_LENGTH,), dtype='int32')
        sentence_embedding = sentence_embedding(sentence_ip)  # Note: these are two different embeddings
        x = LSTM(256, dropout=0.2, recurrent_dropout=0.2)(sentence_embedding)
        x = Dense(3, activation='softmax')(x)
        model = Model(inputs=sentence_ip, outputs=x)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc', f1, precision, recall])
        print(model.summary())
        history = model.fit(sentence_train, y_train, epochs=10, verbose=1, validation_data=(sentence_test, y_test))
        f.write('\nFold %d\n' % (j + 1))
        f.write(str(history.history['acc']))
        f.write(str(history.history['val_acc']))
        f.write(str(history.history['f1']))
        f.write(str(history.history['precision']))
        f.write(str(history.history['recall']))


if __name__ == '__main__':
    model_3()
