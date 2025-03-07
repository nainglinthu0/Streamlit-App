import streamlit as st
import pickle
import numpy as np

# Streamlit UI
st.title("🌸 Iris Flower Prediction App")
st.write("Enter the flower details to predict the species.")

# Input fields for the features
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.0)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=4.0)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=1.5)

# Load the saved model and scaler
with open('knn_iris_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)
    
with open('knn_iris_scaler.pkl', 'rb') as scaler_file:
    loaded_scaler = pickle.load(scaler_file)

# Prediction button
if st.button("Predict"):
    try:
        # Prepare the input features as a 2D numpy array
        input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # Scale the input features using the loaded scaler
        input_features_scaled = loaded_scaler.transform(input_features) 
        
        # Predict the species using the loaded model
        prediction = loaded_model.predict(input_features_scaled)

        # Iris species names
        species_names = ['Setosa', 'Versicolor', 'Virginica']

        # Display the result
        st.write(f"Prediction: The flower is likely a {species_names[prediction[0]]}.")

    except Exception as e:
        st.error(f"Error: {e}")