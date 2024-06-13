from flask import Flask, request, jsonify,render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
model = pickle.load(open('finalized_model.pkl', 'rb'))

"""
@app.route("/")
def welcome():
    return "welcome to flask"
"""
@app.route("/test")
def test():
    return "testing routes"

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/api', methods=['POST'])
def predict():
    # Get the data from the POST request.
    data = request.get_json(force=True)
    print("data ",data)

    # Make prediction using model loaded from disk as per the data.
    prediction = model.predict([[np.array(int(data['area']))]])

    # Take the first value of prediction
    output = prediction[0]
    print("output ",output)

    return jsonify({"data":output[0]})

if __name__ == '__main__':
    app.run(port=5000, debug=True)

if(__name__=='__main__'):
    app.run(host='0.0.0.0', port = 8080)