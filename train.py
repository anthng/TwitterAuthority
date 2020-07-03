import tensorflow_hub as hub
import tensorflow as tf
import bert
from bert import bert_tokenization

import math

from utils import bert_encode
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from tensorflow.keras import backend as K

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import class_weight

base_model = hub.load('https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1')


bert_layer = hub.KerasLayer(base_model)

vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = bert_tokenization.FullTokenizer(vocab_file, do_lower_case)


MAX_SEQ_LEN = 65
TEST_SIZE = 0.2
LR = 1e-4

N_EPOCHS = 5
BATCH_SIZE = 32



df = pd.read_csv('./data/clean_train.csv', index_col = False)
df.dropna(inplace = True)
df.reset_index(drop=True,inplace=True)
print("Split Data")
X_data = df['text'][:5].to_numpy()
y_data = df['target'][:5].to_numpy()
y_data = y_data.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=TEST_SIZE, random_state=42)
class_weights = class_weight.compute_class_weight('balanced', np.unique(df['target'].to_numpy()), df['target'].to_numpy())
class_weights = dict(enumerate(class_weights))
print(class_weights)
    
print("Bert encoding")
X_train = bert_encode(X_train, tokenizer=tokenizer, sequence_length=MAX_SEQ_LEN)
X_test = bert_encode(X_test, tokenizer=tokenizer, sequence_length=MAX_SEQ_LEN)

print("Build Model")

input_word_ids = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32,name="input_ids")
input_mask = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32,name="input_masks")
input_segment = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32,name="segment_ids")
pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, input_segment])
biLstm_output = tf.keras.layers.LSTM(units=32, return_sequences=True)(sequence_output)

avg_pool = tf.keras.layers.GlobalAveragePooling1D()(biLstm_output)
dense = tf.keras.layers.Dense(20, activation='relu')(avg_pool)
pred = tf.keras.layers.Dense(1, activation='sigmoid')(dense)
model = tf.keras.models.Model(inputs=[input_word_ids,input_mask,input_segment], outputs=pred)
model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=LR), metrics=['accuracy'])
model.summary()

print("training")
model.fit(X_train, y_train, class_weight = class_weights, epochs=N_EPOCHS, batch_size=BATCH_SIZE)    

print("Test and Evaluate")
y_prob = model.predict(X_test, verbose = 1)
y_pred = (y_prob > 0.5).astype(int)
print(classification_report(y_test, y_pred, digits = 4))

print("...Save model")
model.save('./models/rnn')