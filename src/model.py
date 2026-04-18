from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dropout, Dense
from config import MAX_DIM_INPUT,MAX_LEN

def build_model(MAX_DIM_INPUT, MAX_LEN):

    model = Sequential()

    # Embedding
    model.add(Embedding(input_dim=MAX_DIM_INPUT, output_dim=48, input_shape=(MAX_LEN,)))

    # RNN Layer 1
    model.add(SimpleRNN(96, return_sequences=True))
    model.add(Dropout(0.2))

    # RNN Layer 2
    model.add(SimpleRNN(80, return_sequences=True))
    model.add(Dropout(0.2))

    # RNN Layer 3
    model.add(SimpleRNN(112))
    model.add(Dropout(0.2))

    # Output
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    return model 

