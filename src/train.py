import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping


from preprocess import preprocess_text
from model import build_model

# Config
MAX_LEN = 250
MAX_DIM_INPUT = 10000 # VOCAB_SIZE
MODEL_PATH = "models/vs_code_train_rnn_model.h5"
TOKENIZER_PATH = "models/vs_code_train_tokenizer.pickle"

# Load data
df = pd.read_csv("data/raw/IMDB Dataset.csv")

# For testing 
df = df.sample(5000) 

texts = df["review"].astype(str)
labels = df["sentiment"]

# Preprocess text
texts = texts.apply(preprocess_text)

# Tokenizer
tokenizer = Tokenizer(num_words=MAX_DIM_INPUT, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

# Save tokenizer 
with open(TOKENIZER_PATH, "wb") as f:
    pickle.dump(tokenizer, f)

# Convert to sequences
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=MAX_LEN)


# Initialize the encoder
le = LabelEncoder()
# Fit and transform your labels
y = le.fit_transform(df['sentiment'])


# Train-test split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)

# Build model
model = build_model(MAX_DIM_INPUT=MAX_DIM_INPUT, MAX_LEN=MAX_LEN)


# Configure the monitor
early_stop = EarlyStopping(
    monitor='val_loss',      # Watch the validation loss
    mode='min',              # We want the loss to be as low as possible
    patience=10,              # Wait for 10 epochs of no improvement before stopping
    restore_best_weights=True # Very important: gives you the best version of the model
)

# Train
model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=64,       # How many reviews the model looks at before updating weights
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

# Save model
model.save(MODEL_PATH)

print(" Model and tokenizer saved!")