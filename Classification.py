import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.utils.np_utils import to_categorical

train_filename = "C:\\Users\\李金强\\Desktop\\工作室数据\\train.csv"
train_df = pd.read_csv(train_filename)
label_df = train_df['label']
label = label_df.values
del train_df['label']
del train_df['id']
X_train = train_df.values.reshape(8001,)
Y_train = to_categorical(label, 10)
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 256
max_len = 200
X_train = pad_sequences(X_train, padding='post', maxlen=max_len)
X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=max_len)

model = Sequential()
model.add(layers.Embedding(input_dim=vocab_size,
                           output_dim=embedding_dim,
                           input_length=max_len))
model.add(layers.Conv1D(64, 5, activation='relu'))
model.add(layers.Conv1D(128, 5, activation='relu'))
model.add(layers.Conv1D(256, 5, activation='relu'))
model.add(layers.Conv1D(512, 5, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(Dropout(0.4))
model.add(layers.Dense(128, activation='relu'))
model.add(Dropout(0.4))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
model.fit(X_train, Y_train,
          batch_size=32, epochs=10, verbose=1)
test_filename = "C:\\Users\\李金强\\Desktop\\工作室数据\\test.csv"
test_df = pd.read_csv(test_filename)
del test_df['id']
test = test_df.values.reshape(20001,)
test = tokenizer.texts_to_sequences(test)
test = pad_sequences(test, padding='post', maxlen=max_len)
test = tf.keras.preprocessing.sequence.pad_sequences(test, maxlen=max_len)
predict = model.predict(test)
predict = np.argmax(predict, axis=1)
id = np.arange(0, 20001)
submission = pd.DataFrame({'id': id, 'label': predict})
submission.to_csv("C:\\Users\\李金强\\Desktop\\Test Classification.csv", index=0)
