from django.shortcuts import render

# Create your views here.
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from rest_framework.decorators import api_view
from rest_framework.response import Response
import joblib
import pandas as pd

@api_view(['GET'])
def predict(request):
    # Load the saved model
    model = joblib.load('api/models/modell.pkl')
    
    # Get input data from the request body
    input_data = request.data
    
    # Extract features from input data
    features = {
        'Kilométrage': input_data.get('Kilométrage'),
        'Année': input_data.get('Année'),
        'MoteurType': input_data.get('MoteurType'),
        'Boite_encoded': input_data.get('Boite_encoded'),
        'Energie_encoded': input_data.get('Energie_encoded'),
        'Modèle_encoded': input_data.get('Modèle_encoded'),
        'Marque_encoded': input_data.get('Marque_encoded'),
    }
    
    # Convert features to pandas DataFrame for prediction
    features_df = pd.DataFrame([features])
    
    # Perform prediction
    prediction = model.predict(features_df)
    
    # Return prediction as JSON response
    return Response({'features_array': features,'prediction':prediction})
