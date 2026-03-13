import matplotlib
matplotlib.use('Agg') #Use non-interactive backend before importing the pyplot

from flask import Flask,request,jsonify,send_file
from flask_cors import CORS
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import mlflow
import numpy as np
import re
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from mlflow.tracking import MlflowClient
import matplotlib.dates as mdates
import pickle


app=Flask(__name__)
CORS(app) #ENable CORS for all routes

def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        #Convert to lower
        comment=comment.lower()

        #Remove trailing and leading whitespaces
        comment=comment.strip()

        # remove newline characters
        comment=re.sub(r'\n',' ',comment)

        #remove non-alphanumeric characters,except punctuation
        comment=re.sub(r'[^A-Za-z0-9\s!?.,]','',comment)

        #remove stopwords but retain important ones for sentiment analysis
        stop_words=set(stopwords.words('english'))-{'not','but','however','no','yet'}
        comment=' '.join([word for word in comment.split() if word not in stop_words])

        #Lemmatize the words
        lemmatizer=WordNetLemmatizer()
        comment=' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        print(f"Error in preprocessing comment: {e}")
        return comment
    
#Load the model and vectorizer from the model registery annd local storage

def load_model_and_vectorizer(model_name,model_version,vectorizer_path):
    #set MLflow tracking URI to your server
    mlflow.set_tracking_uri("") #Replace with your MLflow tracking URI  
    client=MlflowClient()
    model_uri=f"model:/{model_name}/{model_version}"
    model=mlflow.pyfunc.load_model(model_uri)

    with open(vectorizer_path,'rb') as file:
        vectorizer=pickle.load(file)

    return model,vectorizer

## initialize the model and vectorizer
model,vectorizer=load_model_and_vectorizer("my_model","1","./tfidf_vectorizer.pkl") #update the paths and vectorizer


@app.route('/')
def home():
    return "Welcome to our flask api"

@app.route('/predict',methods=['POST'])
def predict():
    data=request.json
    comments=data.get("comments")
    print("I am in the comment:",comments)
    print("I am the comment type: ",type(comments))

    if not comments:
        return jsonify({"error":"No comments provided"}),400
    
    try:
        # Preprocess each comment before vectorizing
        preprocessed_comments=[preprocess_comment(comment) for comment in comments]

        #Transform comments using the cvectorizing
        transformed_comments=vectorizer.transform(preprocess_comment)

        #Convert the sparse matrix to dense format
        dense_comments=transformed_comments.toarray() #Convert to dense array

        #Make predictions
        predictions=model.predict(dense_comments).tolist() #Conver to list


    except Exception as e:
        return jsonify({"error":f"Prediction failed: {str(e)}"}),500
    
    response=[{"comment":comment,"sentiment":sentiment} for comment,sentiment in zip(comments,predictions)]
    return jsonify(response)

if __name__=='__main__':
    app.run(host='0.0.0.0',port=5000,debug=True)