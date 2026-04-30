import os
import pandas as pd
import joblib

from src.data_preprocessing import load_data, preprocess_data
from src.train_model import train_model

# Create folders
os.makedirs('models', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# Load data
df = load_data('data/student_data.csv')

# Preprocess
df = preprocess_data(df)

X = df.drop('final_score', axis=1)
y = df['final_score']

# Train
model = train_model(X, y)

# Save model
joblib.dump(model, 'models/student_model.pkl')

# Predict
preds = model.predict(X)

# Save predictions
pd.DataFrame(preds, columns=['Predictions']).to_csv('outputs/predictions.csv', index=False)

print("Project executed successfully!")