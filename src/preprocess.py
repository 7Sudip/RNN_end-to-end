import re
import string
import pickle
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.config import MAX_LEN,TOKENIZER_PATH

import streamlit as st

# Setup resources
nltk.download('punkt',quiet=True)
nltk.download('punkt_tab',quiet=True)
nltk.download('stopwords',quiet=True)
nltk.download('wordnet',quiet=True)

# Load tokenizer
@st.cache_resource # For fast in Streamlit UI
def load_tokenizer():
    with open(TOKENIZER_PATH, "rb") as handle:
        tokenizer = pickle.load(handle)
    return tokenizer

tokenizer = load_tokenizer() 


exclude = string.punctuation
lemmatizer = WordNetLemmatizer()

# Your slang_map
slang_map = {
    "dont": "do not", "im": "i am", "doesnt": "does not", "didnt": "did not",
    "thats": "that is", "ive": "i have", "isnt": "is not", "theres": "there is",
    "wasnt": "was not", "youre": "you are", "couldnt": "could not", "youll": "you will",
    "theyre": "they are", "wouldnt": "would not", "arent": "are not", "havent": "have not",
    "youve": "you have", "whos": "who is", "werent": "were not", "youd": "you would",
    "hasnt": "has not", "shouldnt": "should not", "hadnt": "had not", "weve": "we have",
    "aint": "am not", "wouldve": "would have", "couldve": "could have", "theyd": "they would",
    "theyve": "they have", "theyll": "they will", "hed": "he would","cant":"can not",
    "tv": "television", "dvd": "digital video disc","vhs": "video home system", "scifi": "science fiction",
    "cgi": "computer generated imagery",
    "imdb": "internet movie database", "hbo": "home box office",
    "cia": "central intelligence agency",
    "wwii": "world war two", "bbc": "british broadcasting corporation",
    "80s": "eighties", "70s": "seventies", "60s": "sixties", "50s": "fifties",
    "40s": "forties", "30s": "thirties", "90s": "nineties"
}

# All listed stopwords
stop_word_list = stopwords.words('english')

# Define words that are actually IMPORTANT for sentiment
keep_words = {'not', 'no', 'never', 'but', 'very', 'too', 'so', 'really', 'however', 'nor'}

# Create your custom list
custom_stopwords = [word for word in stop_word_list if word not in keep_words]


def preprocess_text(text):

    # 1. Lowercase
    text = text.lower()

    # 2. Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # 3. Handle non-ASCII
    text = text.encode("ascii", "ignore").decode()
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)

    # 4. Remove punctuation
    for char in exclude:
        text = text.replace(char, '')

    # 5. Handle slang + corrections
    for word, replacement in slang_map.items():
        pattern = r'\b' + word + r'\b'
        text = re.sub(pattern, replacement, text)

    text = re.sub(r'\s+', ' ', text).strip()

    # 6. Handle garbage / normalization
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    words = text.split()
    clean_words = []
    save_list = {'ii', 'iii', 'iv'}

    for w in words:
        if w in ['a', 'i'] or w in save_list:
            clean_words.append(w)
            continue

        is_mash = all(c == w[0] for c in w)
        if len(w) > 1 and not is_mash:
            clean_words.append(w)

    text = " ".join(clean_words)

    # 7. Remove custom stopwords
    words = text.split()
    words = [w for w in words if w not in custom_stopwords]
    text = " ".join(words)

    # 8. Lemmatization (verb-based)
    words = text.split()
    text = " ".join([lemmatizer.lemmatize(w, pos=wordnet.VERB) for w in words])

    return text


def text_to_sequence(input_text):
    processed_text = preprocess_text(input_text)
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded = pad_sequences(sequence, maxlen=MAX_LEN)
    return padded 
