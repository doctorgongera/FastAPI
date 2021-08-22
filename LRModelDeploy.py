from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
import numpy as np
from fastapi import FastAPI, Form
import pandas as pd
from starlette.responses import HTMLResponse 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import re

def preProcess_data(text): #cleaning the data
    
    text = text.lower()
    new_text = re.sub('[^a-zA-z0-9\s]','',text)
    new_text = re.sub('rt', '', new_text)
    return new_text



data = pd.read_csv('tweet_train.csv')
data = data[['text','sentiment']]
data = data.dropna()
tokenizer = Tokenizer(num_words=2000, split=' ')
tokenizer.fit_on_texts(data['text'].values)



def my_pipeline(text): #pipeline
  text_new = preProcess_data(text)
  X = tokenizer.texts_to_sequences(pd.Series(text_new).values)
  X = pad_sequences(X, maxlen=28)
  return X

app = FastAPI()

@app.get('/') #basic get view
def basic_view():
    return {"WELCOME": "GO TO /docs route, or /post or send post request to /predict "}



@app.get('/predict', response_class=HTMLResponse) #data input by forms
def take_inp():
    return '''<form method="post" align="center"> 
    <legend style="font-size:30px">Sentiment Prediction </legend><br><br>
    <input  style="font-size:20px;" size="80" type="text" maxlength="280" name="text" value="Enter Your Text..."/><br><br>
    <input type="submit"/> 
    </form>'''


@app.post('/predict') #prediction on data
def predict(text:str = Form(...)): #input is from forms
    clean_text = my_pipeline(text) #cleaning and preprocessing of the texts
    #loaded_model = tf.keras.models.load_model('TwitterAnalysisDL.h5') #loading the saved model
    #Added Lines
    #Load Models
    model1 = load_model('TwitterAnalysisRNN.h5')
    model2 = load_model('TwitterAnalysisLSTM.h5')
    ensembleModel = [model1, model2]
    ## Predictions 
    prediction=[model.predict(clean_text) for model in ensembleModel]
    prediction=np.array(prediction)
    prediction=np.sum(prediction, axis=0) #making predictions
    sentiment = int(np.argmax(prediction)) #index of maximum prediction
    probability = max(prediction.tolist()[0]) #probability of maximum prediction
    if sentiment==1: 
        t_sentiment = 'Negative'
    elif sentiment==0:
        t_sentiment = 'Neutral'
    elif sentiment==2:
        t_sentiment='Positive'
    
    return { #returning a dictionary as endpoint
        "ACTUALL SENTENCE": text,
        "PREDICTED SENTIMENT": t_sentiment,
        "Probability": probability*0.5
    
    }