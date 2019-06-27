import os
import re
import sys
from collections import OrderedDict, defaultdict

import numpy as np
from keras.callbacks import EarlyStopping
from keras.initializers import Constant
from keras.layers import (GRU, LSTM, RNN, Bidirectional, Concatenate, Conv1D,
                          Dense, Dropout, Embedding, Flatten,
                          GlobalMaxPooling1D, Input, MaxPooling1D, Reshape)
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from nltk.tokenize import TweetTokenizer
from pandas import read_csv
from sklearn.metrics import f1_score


class CustomTokenizer():
    def __init__(self, num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ', oov_token=None):
        self.tokenizer = TweetTokenizer()
        self.word_counts = OrderedDict()
        self.word_index = dict()
        self.index_word = dict()
        self.word_docs = defaultdict(int)
        self.num_words = num_words
        self.filters = filters
        self.lower = lower
        self.split = split
        self.oov_token = oov_token

    def fit_on_texts(self, texts):
        for text in texts:
            seq = self.text_to_word_sequence(
                text, self.filters, self.lower, self.split)
            for w in seq:
                if w in self.word_counts:
                    self.word_counts[w] += 1
                else:
                    self.word_counts[w] = 1
            for w in set(seq):
                # In how many documents each word occurs
                self.word_docs[w] += 1

        wcounts = list(self.word_counts.items())
        wcounts.sort(key=lambda x: x[1], reverse=True)
        # forcing the oov_token to index 1 if it exists
        if self.oov_token is None:
            sorted_voc = []
        else:
            sorted_voc = [self.oov_token]
        sorted_voc.extend(wc[0] for wc in wcounts)

        # note that index 0 is reserved, never assigned to an existing word
        self.word_index = dict(
            list(zip(sorted_voc, list(range(1, len(sorted_voc) + 1)))))

        self.index_word = dict((c, w) for w, c in self.word_index.items())

    def text_to_word_sequence(self, text,
                              filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                              lower=True, split=' '):
        """Converts a text to a sequence of words (or tokens).

        # Arguments
            text: Input text (string).
            filters: list (or concatenation) of characters to filter out, such as
                punctuation. Default: ``!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n``,
                includes basic punctuation, tabs, and newlines.
            lower: boolean. Whether to convert the input to lowercase.
            split: str. Separator for word splitting.

        # Returns
            A list of words (or tokens).
        """
        text = re.sub(r'@USER|URL', split, text)

        if lower:
            text = text.lower()

        text = re.sub(r'bi\*ch|b\*\*ch|bi\*\*h|biatch', 'bitch', text)
        text = re.sub(r'sob|sobi*ch', 'son of bitch', text)
        text = re.sub(r'f\*\*k|f\*ck|fu\*k', 'fuck', text)
        text = re.sub(r'[\'’]s', ' is', text)
        text = re.sub(r'[\'’]re', ' are', text)

        translate_dict = dict((c, split) for c in filters)
        translate_map = str.maketrans(translate_dict)
        text = text.translate(translate_map)

        seq = self.tokenizer.tokenize(text)

        return [i for i in seq if i]

    def texts_to_sequences_generator(self, texts):
        """Transforms each text in `texts` to a sequence of integers.

        Each item in texts can also be a list,
        in which case we assume each item of that list to be a token.

        Only top "num_words" most frequent words will be taken into account.
        Only words known by the tokenizer will be taken into account.

        # Arguments
            texts: A list of texts (strings).

        # Yields
            Yields individual sequences.
        """
        num_words = self.num_words
        oov_token_index = self.word_index.get(self.oov_token)
        for text in texts:
            seq = self.text_to_word_sequence(text,
                                             self.filters,
                                             self.lower,
                                             self.split)
            vect = []
            for w in seq:
                i = self.word_index.get(w)
                if i is not None:
                    if num_words and i >= num_words:
                        if oov_token_index is not None:
                            vect.append(oov_token_index)
                    else:
                        vect.append(i)
                elif self.oov_token is not None:
                    vect.append(oov_token_index)
            yield vect

    def texts_to_sequences(self, texts):
        """Transforms each text in texts to a sequence of integers.

        Only top "num_words" most frequent words will be taken into account.
        Only words known by the tokenizer will be taken into account.

        # Arguments
            texts: A list of texts (strings).

        # Returns
            A list of sequences.
        """
        return list(self.texts_to_sequences_generator(texts))


TRAIN_FILE = sys.argv[0]
TEST_FILE = sys.argv[1]
GOLD_FILE = sys.argv[2]
GLOVE_DIR = sys.argv[3]
MODEL = sys.argv[4]

df = read_csv(TRAIN_FILE, sep='\t')

texts = []
labels = {
    'a': [],
    'b': [],
    'c': []
}

for index, row in df.iterrows():
    texts.append(row['tweet'])
    labels['a'].append(row['subtask_a'])
    labels['b'].append(row['subtask_b'])
    labels['c'].append(row['subtask_c'])


label_index = {
    'NOT': 0,
    'OFF': 1,
    'UNT': 0,
    'TIN': 1,
    'IND': 0,
    'GRP': 1,
    'OTH': 2
}

label_index = defaultdict(bool, label_index)

# Data preprocessing

# Map label from str to int
for subtask in labels:
    labels[subtask] = list(map(lambda x: label_index[x], labels[subtask]))

MAX_SEQUENCE_LENGTH = 200
MAX_NUM_WORDS = 30000
EMBEDDING_DIM = 200
VALIDATION_SPLIT = 0.2

print('Indexing word vectors.')

embeddings_index = {}
with open(os.path.join(GLOVE_DIR, 'glove.twitter.27B.200d.txt')) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, 'f', sep=' ')
        embeddings_index[word] = coefs

print('Found %s word vectors.' % len(embeddings_index))

# finally, vectorize the text samples into a 2D integer tensor
tokenizer = CustomTokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels_a = to_categorical(np.asarray(labels['a']))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels_a.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels_a = labels_a[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-num_validation_samples]
y_train = labels_a[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels_a[-num_validation_samples:]

print('Preparing embedding matrix.')

# prepare embedding matrix
num_words = min(MAX_NUM_WORDS, len(word_index)) + 1
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i > MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

print(embedding_matrix.shape)

# Model

if MODEL == 'CNN':
    print('Training CNN model.')

    # train a 1D convnet with global maxpooling
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(2, activation='softmax')(x)
elif MODEL == 'RNN':
    print('Training LSTM model.')

    # train a 1D convnet with global maxpooling
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Bidirectional(LSTM(128, return_sequences=True))(embedded_sequences)
    x = Bidirectional(LSTM(128))(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(2, activation='softmax')(x)

earlystopping = EarlyStopping(patience=2)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])
model.summary()

model.fit(x_train, y_train,
          batch_size=128,
          epochs=10,
          validation_data=(x_val, y_val),
          callbacks=[earlystopping])


# Pedict

df_test = read_csv(TEST_FILE, sep='\t')
sequences = tokenizer.texts_to_sequences(df_test['tweet'])
x_test = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
y_pred = model.predict(x_test, verbose=1)
y_pred = np.argmax(y_pred, axis=1)

# Evaluation

df_gold = read_csv(GOLD_FILE, header=None)
y_true = list(map(lambda x: label_index[x], df_gold[1]))
print('f1-score:', f1_score(y_true, y_pred, average='macro'))
