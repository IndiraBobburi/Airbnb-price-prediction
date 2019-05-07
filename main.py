from flask import Flask, render_template
from flask import request, redirect, make_response
# from flask.ext.responses import json_response
from flask_cors import CORS
import os
import os.path
import json
import prediction as obj
app = Flask(__name__, static_url_path = "", static_folder = "tmp")
CORS(app)

#initialize the data and model

@app.route("/")
def first_page():
    return render_template("index.html")
    
@app.route('/predict', methods=['POST'])
def predict():
     data = request.get_json()
     print(data[0])

     # val = obj.predictResult(data)
     # r = {"result":val}
     # # result = "Slight"
     # resp = make_response(json.dumps(r))
     # resp.status_code = 200
     # resp.headers['Access-Control-Allow-Origin'] = '*'
     
     return "SAdsadsa"

@app.route('/handle_data', methods=['POST'])
def handle_data():
    arrayData = []
    arrayData.append('0.0');
    neighbourhood = request.form['neighbourhood']   
    print(neighbourhood);
    arrayData.append(neighbourhood)
    bedrooms = request.form['bedrooms']   
    print(bedrooms);
    arrayData.append(bedrooms)
    houseType = request.form['house-type']   
    print(houseType);
    arrayData.append(houseType)
    roomType = request.form['room-type']   
    print(roomType);
    arrayData.append(roomType)
    name = request.form['name']   
    print(name);
    arrayData.append(name)
    summary = request.form['summary']   
    print(summary);
    arrayData.append(summary)
    array = request.form.getlist('checkboxArray[]');  
    #arrayData.append(array) 
    print("asdasdasdsa  ");
    ame = '{'+  ",".join(map(str,array)) + "}"
    arrayData.append(ame) 
    arrayData.append(62.28261879577949)
    arrayData.append(-71.13306792912681)
    arrayData.append(0)
    arrayData.append('f')
    arrayData.append(2)
    final = [];
    final.append(arrayData)
    imouto = obj.prediction()
    val = imouto.predictResult(final)
    return render_template("result.html", answer = val);  

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=80, debug=False, threaded=True)
