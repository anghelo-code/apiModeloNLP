import spacy
from tqdm import tqdm
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import joblib
from keras.preprocessing import sequence

# Cargar el modelo de spaCy en español
nlp = spacy.load("es_core_news_sm")

palabras_a_conservar = [
    'bien',
    'mal',
    'mejor',
    'peor',
    'gran',
    'más',
    'menos',
    'siempre',
    'nunca',
    'si',
    'no'
]

nlp.Defaults.stop_words.add("https")

# Elimina las palabras de las stop words en spaCy
for palabra in palabras_a_conservar:
  if palabra in nlp.Defaults.stop_words:
    nlp.Defaults.stop_words.remove(palabra)

# Cargar el tokenizador desde el archivo
with open('./tokenizer.pkl', 'rb') as f:
  tokenizer = pickle.load(f)

def tokenizar(text):
  sequences = tokenizer.texts_to_sequences([text])
  word_index = tokenizer.word_index
  return sequence.pad_sequences(sequences, maxlen=400, padding = 'post')

 

def removePunctuation(text):
  # Procesar el texto completo y eliminar signos de puntuación
  doc = nlp(text)
  return [token.text for token in doc if not token.is_punct]

def lowercase(words):
  return [word.lower() for word in words]

def removeStopWords(words):
  doc = nlp(" ".join(words))
  return [word.text for word in doc if not word.is_stop]

def removeShortWords(words, numLetras=3):
  return [word for word in words if len(word) > numLetras]

def lemmatizar(words):
  doc = nlp(" ".join(words))
  return [token.lemma_ for token in doc] 



def procesar1(text, numLetras=3):
  words = lowercase(removePunctuation(text))                         # Convertimos todo a minúsculas y eliminamos signos de puntuación
  words = removeShortWords(words, numLetras)       # Eliminamos palabras cortas
  words = removeStopWords(words)                   # Eliminamos palabras vacías
  words = lemmatizar(words)                        # Lematizamos
  words = tokenizar(" ".join(words))               # Vectorizamos
  return words  # Devuelve el resultado como una cadena de texto

