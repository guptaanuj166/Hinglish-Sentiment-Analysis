fl=1
import streamlit as st
import torch
from torchtext.legacy.data import Field, TabularDataset, BucketIterator
import gensim.models as gm
import pandas as pd
import numpy as np

import numpy as np
from matplotlib import pyplot as plt
import re
import string
from sklearn.metrics import f1_score, confusion_matrix
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from nltk import download
download('stopwords')

import smart_open
smart_open.open = smart_open.smart_open
import pandas as pd
import preprocessor as p
import gensim.models as gm
import time
import random
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns


import torch 
import torchtext
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext.legacy.data import Field, TabularDataset, BucketIterator

from torch.utils.data import DataLoader, Dataset

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# from using_vecmap import LSTM  # Assuming your LSTM class is saved in model.py


# To store the contents of a tweet in an efficient manner
class Data:
    def __init__(self):
        self.uid = None
        self.content = ''
        self.sentiment = ''

# Function to clean (/pre process) a given tweet
def cleanTweet(data):
    # Doing various pre processing steps to clean the contents of the given tweet
    data.content = re.sub(r'\_', '', data.content) # remove underscores
    data.content = re.sub(r'…', '', data.content) # remove elipses/dots
    data.content = re.sub(r'\.', '', data.content) # remove elipses/dots
    data.content = re.sub(r'^RT[\s]+', '', data.content) # remove retweets
    data.content = re.sub("[#@©àâ€¦¥°¤ð¹ÿœ¾¨‡†§‹²¿¸ˆ]", '', data.content) # remove weird symbols
    data.content = data.content.split("http")[0].split('https')[0] # remove http/https
    data.content = ''.join([i for i in data.content if not i.isdigit()]) # remove digits
    data.content = ''.join([word for word in data.content if word not in string.punctuation]) # remove punctuations
    data.content = TweetTokenizer(preserve_case=False, strip_handles=True,reduce_len=True).tokenize(data.content)
    data.content = ' '.join([i for i in data.content]) # convert to string
    return data

# Loading the stopwords form both engish as well as hinglish
#Stop words English: is, and, has, the, etc.
#Hinglish Stop words: aise, abbey, kya, etc.
def load_stop_words():
    stopwords_english = stopwords.words('english')
    stopwords_hinglish = []
    with open('./stop_hinglish.txt','r') as fp:
        while True:
            line = fp.readline()
            if not line:
                break
            stopwords_hinglish.append(line.strip())
    return stopwords_english, stopwords_hinglish



class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout):
        """
        Define the layers of the module.

        vocab_size - vocabulary size
        embedding_dim - size of the dense word vectors
        hidden_dim - size of the hidden states
        output_dim - number of classes
        n_layers - number of multi-layer RNN
        bidirectional - boolean - use both directions of LSTM
        dropout - dropout probability
        pad_idx -  string representing the pad token
        """
        
        super().__init__()

        # 1. Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # 2. LSTM layer
        self.encoder = nn.LSTM(embedding_dim, 
                               hidden_dim, 
                               num_layers=n_layers,
                               bidirectional=bidirectional,
                               dropout=dropout)
        
        # 3. Fully-connected layer
        self.predictor = nn.Linear(hidden_dim * 2, output_dim)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
      
        # 4. Softmax layer for probabilities
        self.softmax = nn.Softmax(dim=1)


    def forward(self, text, text_lengths):
        """
        The forward method is called when data is fed into the model.

        text - [sentence length, batch size]
        text_lengths - lengths of sentences
        """

        # Embedding
        embedded = self.dropout(self.embedding(text))    

        # Pack the embeddings for variable-length sequences
        text_lengths = text_lengths.cpu()
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)

        # LSTM encoder
        packed_output, (hidden, cell) = self.encoder(packed_embedded)

        # Unpack sequences
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
 
        # Concatenate the final forward and backward hidden states and apply dropout
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))

        # Pass through the fully connected layer
        logits = self.predictor(hidden)  # [batch size, output_dim]

        print(logits)
        # Apply softmax to get probabilities
        # probabilities = self.softmax(logits)  # [batch size, output_dim]
        # print(probabilities)
        # outputs = logits
        # max_probab, _ = torch.max(outputs,1)
        return logits

# Define constants
INPUT_DIM = 37195  # Adjust based on your vocab size
EMBEDDING_DIM = 25
HIDDEN_DIM = 128
OUTPUT_DIM = 3
N_LAYERS = 4
BIDIRECTIONAL = True
DROPOUT = 0.5


# Load resources
@st.cache_resource
def load_resources():
    TEXT = Field(lower=True, include_lengths=True)
    LABEL = Field(sequential=False, use_vocab=False, dtype=torch.float)
    fields = [('id', None), ('text', TEXT), ('label', LABEL)]
    dataset = TabularDataset(
        path="train_data.csv", format="CSV", fields=fields, skip_header=True
    )
    TEXT.build_vocab(dataset)
    vocab = TEXT.vocab
    return TEXT, vocab

TEXT, vocab = load_resources()

# Load pre-trained embeddings
@st.cache_resource
def load_embeddings():
    model_pretrained = gm.KeyedVectors.load_word2vec_format('model_trgt.model')
    embedding_matrix = torch.zeros(len(vocab), EMBEDDING_DIM)
    for word, idx in vocab.stoi.items():
        if word in model_pretrained.key_to_index:
            embedding_matrix[idx] = torch.tensor(model_pretrained[word])
        else:
            embedding_matrix[idx] = torch.ones(EMBEDDING_DIM)
    return embedding_matrix

embedding_matrix = load_embeddings()

# Initialize model
@st.cache_resource
def load_model():
    model = LSTM(
        vocab_size=len(vocab),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        n_layers=N_LAYERS,
        bidirectional=BIDIRECTIONAL,
        dropout=DROPOUT
    )
    model.embedding.weight.data.copy_(embedding_matrix)
    model.load_state_dict(torch.load('model-best.pth'))
    model.eval()
    return model

model = load_model()

# Preprocessing function
def preprocess_text(text):
    cleaned_text = text.lower().replace("\n", " ")
    tokenized = [token for token in cleaned_text.split() if token in vocab.stoi]
    indexed = [vocab.stoi[token] for token in tokenized]
    return torch.tensor(indexed, dtype=torch.long).unsqueeze(1), torch.tensor(len(indexed))

# Prediction function
def predict(model, text):
    text_tensor, text_length = preprocess_text(text)
    with torch.no_grad():
        predictions = model(text_tensor, text_length.unsqueeze(0))
        probs = torch.softmax(predictions, dim=1).squeeze()
    return probs.numpy()

# Streamlit GUI
st.title("Senti Hinglish")

input_mode = st.radio("Choose Input Mode", ["Text Input", "File Upload"])

if input_mode == "Text Input":
    user_input = st.text_area("Enter your text below:")
    if st.button("Analyze"):
        if user_input.strip():
            probabilities = predict(model, user_input)
            st.write(user_input)
            labels = ['Negative', 'Neutral', 'Positive']
            label = np.argmax(probabilities)
            if fl==0:
            # Map the index to the corresponding label
                  # Assuming index 0: Negative, 1: Neutral, 2: Positive
                if(probabilities[2]>probabilities[0]):
                    st.write(f"Positive")
                else:
                    st.write(f"Negative")
                    
            else:
                st.write(f"{labels[label]}")
                # st.write(f"Neutral")
            # Display the label with the maximum probability
            # st.write(f"Predicted Sentiment: {labels[label]}")
            # # st.write("Sentiment Probabilities:")
            # st.write(f"Neutral: {probabilities[1]:.2f}")
            # st.write(f"Positive: {probabilities[2]:.2f}")
            # st.write(f"Negative: {probabilities[0]:.2f}")
        else:
            st.warning("Please enter text for analysis.")
else:
    uploaded_file = st.file_uploader("Upload a CSV file with a 'text' column", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if "text" in df.columns:
            df['Sentiment'] = df['text'].apply(lambda x: np.argmax(predict(model, x)))
            st.write("Predictions:")
            st.dataframe(df)
            st.download_button(
                "Download Predictions",
                df.to_csv(index=False).encode("utf-8"),
                "predictions.csv",
                "text/csv",
                key="download-csv"
            )
        else:
            st.error("The uploaded file must contain a 'text' column.")
