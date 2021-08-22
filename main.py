from fastapi import FastAPI, Form
from starlette.responses import HTMLResponse
import LRModelDeploy.ipynb

app = FastAPI()

@app.get('/') #basic get view
def basic_view():
    return {"WELCOME": "GO TO /docs route, or /post or send post request to /predict "}



@app.get('/predict', response_class=HTMLResponse) #data input by forms
def take_inp():
    return '''<form method="post"> 
    <input type="text" maxlength="28" name="text" value="Text Emotion to be tested"/>  
    <input type="submit"/> 
    </form>'''



@app.post('/predict') #prediction on data
def predict(text:str = Form(...)): #input is from forms
    clean_text = my_pipeline(text) #cleaning and preprocessing of the texts
    loaded_model = tf.keras.models.load_model('sentiment.h5') #loading the saved model
    predictions = loaded_model.predict(clean_text) #making predictions
    sentiment = int(np.argmax(predictions)) #index of maximum prediction
    probability = max(predictions.tolist()[0]) #probability of maximum prediction
    if sentiment==0: #assigning appropriate name to prediction
        t_sentiment = 'negative'
    elif sentiment==1:
        t_sentiment = 'neutral'
    elif sentiment==2:
        t_sentiment='postive'
    
    return { #returning a dictionary as endpoint
        "ACTUALL SENTENCE": text,
        "PREDICTED SENTIMENT": t_sentiment,
        "Probability": probability
    }