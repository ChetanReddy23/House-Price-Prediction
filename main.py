from flask import Flask,render_template,request
import pandas as pd
import pickle

app=Flask(__name__)
data=pd.read_csv('Cleaned_data.csv')
pipe=pickle.load(open("model.pkl",'rb'))

@app.route('/')
def index():
    locations=sorted(data['location'].unique())
    Areas=sorted(data['area_type'].unique())
    avail=sorted(data['availability'].unique())
    return render_template('home1.html', location=locations,Area=Areas,Availability=avail)

@app.route('/predict',methods=['POST'])
def predict():

    location =request.form.get('location')
    area_type=request.form.get('Area')
    availability=request.form.get('Availability')
    bhk=request.form.get('bhk')
    bath=request.form.get('bath')
    sqft=request.form.get('sqft')
    
    input=pd.DataFrame([[sqft,bhk,bath,availability, area_type, location]],columns=['sqft', 'bhk', 'bath', 'availability', 'area_type', 'location'])

    prediction =round(pipe.predict(input)[0] * 1e5)
    return str(prediction)
    
    #print(sqft,bhk,bath,availability,area_type,location)
    #return ""

if __name__=="__main__":
    app.run(debug=True,port=5001)
