from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

@app.route('/',methods=["GET","POST"])
def home():
    message = "Welcome to my first flask based web application ... !!!"
    return render_template("home.html", message = message)

@app.route('/getResponseLinearReg',methods=["GET","POST"])
def getResponseLinearReg():
    CRIM = request.form["CRIM"]
    ZN = request.form["ZN"]
    INDUS = request.form["INDUS"]
    CHAS = request.form["CHAS"]
    NOX = request.form["NOX"]
    RM = request.form["RM"]
    AGE = request.form["AGE"]
    DIS = request.form["DIS"]
    RAD = request.form["RAD"]
    TAX = request.form["TAX"]
    PT = request.form["PT"]
    B = request.form["B"]
    LSTAT = request.form["LSTAT"]
    inputList = [CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PT,B,LSTAT]
    with open("boston_mlm.pkl", 'rb') as file:
            pickle_model = pickle.load(file)
            y_pred_from_pkl = pickle_model.predict([inputList])
    print(y_pred_from_pkl)
    return str(y_pred_from_pkl[0])

if __name__ == '__main__':
    app.run(debug=True)