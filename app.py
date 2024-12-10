import numpy as np
from flask import Flask , request , render_template
import pickle
from sklearn.preprocessing import StandardScaler
import pandas as pd

df=pd.read_csv('lib/data.csv')
x=df.drop(columns='Outcome',axis=1)

scaler=StandardScaler()
scaler.fit(x)

app = Flask(__name__)
model = pickle.load(open('model/model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    data_array = np.asarray(int_features)
    data_reshaped = data_array.reshape(1,-1)
    std_data=scaler.transform(data_reshaped)
    prediction = model.predict(std_data)
    print(prediction[0])
    if prediction[0]==1:
        return render_template('index.html', prediction_text='Person is Likely Diabetic')
    else:
         return render_template('index.html', prediction_text='Person is Likely not Diabetic')
    

if __name__ == "__main__":
    app.run()