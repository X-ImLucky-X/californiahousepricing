import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__) #this is the starting point of our application
regmodel=pickle.load(open('regmodel.pkl','rb'))#load the model
scaler=pickle.load(open('scaling.pkl','rb'))#load the scaler

#this will take to the home page
@app.route('/')
def home():
    return render_template('home.html') #this is our html page

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.json['data'] #as soon as we get the data from the frontend it will be in json format and will be captured in data variable
    print(data)
    input_data = np.array(list(data.values())).reshape(1, -1) #convert the data into numpy array and reshape it to 2D array
    print(input_data)
    input_data = scaler.transform(input_data) #apply the same scaling that we did while training the model
    output = regmodel.predict(input_data)
    print(output[0])
    return jsonify(output[0])

if __name__ == "__main__":
    app.run(debug=True) #debug=True will help us to see the errors in the console