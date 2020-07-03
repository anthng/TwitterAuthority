import pandas as pd
import numpy as np

import bert
from bert import bert_tokenization

import tensorflow_hub as hub
import tensorflow as tf

import pandas as pd
import numpy as np

from utils import bert_encode


MAX_SEQ_LEN = 65

base_model = hub.load('https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1')
model_path = './models/'

bert_layer = hub.KerasLayer(base_model)

vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = bert_tokenization.FullTokenizer(vocab_file, do_lower_case)

df = pd.read_csv('./data/test_clean.csv', index_col = False)
df['text'] = df['text'].fillna("")
X_data = df['text'].to_numpy()

X_encode = bert_encode(X_data, tokenizer=tokenizer, sequence_length=MAX_SEQ_LEN)

model = tf.keras.models.load_model('./models/rnn', custom_objects=None, compile=True)
y_prob = model.predict(X_encode, verbose = 1)
y_pred = (y_prob > 0.5).astype(int)

submission = pd.read_csv('./data/sample_submit.csv', index_col = False)

submission["target"] = y_pred
submission.to_csv("./results/result.csv", index = False)