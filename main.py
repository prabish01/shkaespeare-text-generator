import random
import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
from keras.optimizers import RMSprop


filepath = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(filepath,'rb').read().decode(encoding='utf-8').lower()

text = text[300000:800000]  # Truncate text for faster training

characters = sorted(set(text))
char_to_index = dict((c,i) for  i,c in enumerate(characters))
index_to_char = dict((i,c) for  i,c in enumerate(characters))

seq_length = 40
step = 3

sentence = []
next_characters =[]


# for i in range(0,len(text) - seq_length, step):
#     sentence.append(text[i:i + seq_length])
#     next_characters.append(text[i + seq_length])
    
# x = np.zeros((len(sentence), seq_length, len(characters)), dtype=bool)
# y = np.zeros((len(sentence), len(characters)), dtype=bool)

# for i, sentence in enumerate(sentence):
#     for t, character in enumerate(sentence):
#         x[i,t, char_to_index[character]] = 1
#     y[i, char_to_index[next_characters[i]]] = 1
    
# model = Sequential()
# model.add(tf.keras.layers. Input(shape=(seq_length, len(characters))))
# model.add(LSTM(128))
# model.add(Dense(len(characters)))
# model.add(Activation('softmax'))

# optimizer = RMSprop(learning_rate=0.01)
# model.compile(loss='categorical_crossentropy', optimizer=optimizer)

# model.fit(x,y,batch_size=256, epochs=10)

# model.save('text_generator.keras')

model = tf.keras.models.load_model('text_generator.keras')

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text( length=400, temperature=1.0):
    start_index = random.randint(0, len(text) - seq_length - 1)
    generated = ''
    sentence = text[start_index: start_index + seq_length]
    generated += sentence

    for i in range(length):
        x_pred = np.zeros((1, seq_length, len(characters)))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_to_index[char]] = 1

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, temperature)
        next_char = index_to_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

    return generated
print("Generating text with different temperatures:")
print("----------------0.2----------------")
print(generate_text(400, temperature=0.2))
print("----------------0.3----------------")
print(generate_text(400, temperature=0.3))
print("----------------0.4---------------")
print(generate_text(400, temperature=0.4))
print("----------------0.5---------------")
print(generate_text(400, temperature=0.5))
print("----------------0.6---------------")
print(generate_text(400, temperature=0.6))
print("----------------0.7---------------")
print(generate_text(400, temperature=0.7))
print("----------------0.8---------------")
print(generate_text(400, temperature=0.8))
print("----------------0.9---------------")
print(generate_text(400, temperature=0.9))
print("----------------1.0---------------")
print(generate_text(400, temperature=1.0))
