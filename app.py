from flask import Flask, request, render_template
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load the model and scalers
model = pickle.load(open(r'C:\Users\KINJAL MEHRA\OneDrive\Desktop\DATA SCIENCE 2024\Project\model.pkl', 'rb'))
sc = pickle.load(open(r'C:\Users\KINJAL MEHRA\OneDrive\Desktop\DATA SCIENCE 2024\Project\standscaler.pkl', 'rb'))
ms = pickle.load(open(r'C:\Users\KINJAL MEHRA\OneDrive\Desktop\DATA SCIENCE 2024\Project\minmaxscaler.pkl', 'rb'))

# Create Flask app
app = Flask(__name__, static_folder='static')

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data
    N = int(request.form['Nitrogen'])
    P = int(request.form['Phosphorus'])
    K = int(request.form['Potassium'])
    temp = float(request.form['Temperature'])
    humidity = float(request.form['Humidity'])
    ph = float(request.form['Ph'])
    rainfall = float(request.form['Rainfall'])

    # Prepare feature list and transform
    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    # Scale the features
    scaled_features = ms.transform(single_pred)
    final_features = sc.transform(scaled_features)

    # Predict
    prediction = model.predict(final_features)

    # Map prediction to crop name
    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

    # Get result
    result = crop_dict.get(prediction[0], "Sorry, we could not determine the best crop to be cultivated with the provided data.")

    return render_template('index.html', result=result)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)