import os
import requests
import zipfile
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import joblib

# URL of the zip file
url = 'https://cricsheet.org/downloads/t20s_json.zip'
zip_path = 't20s_json.zip'
extract_path = 't20s_json/'

response = requests.get(url)
with open(zip_path, 'wb') as f:
    f.write(response.content)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# Function to load a single JSON file and extract match-level data
def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        match_info = data['info']
        outcome = match_info.get('outcome', {})

        if 'winner' in outcome:
            winner = outcome['winner']
        else:
            return None  # Skip matches with no outcome recorded

        toss_winner = match_info['toss']['winner']
        team1, team2 = match_info['teams']
        gender = match_info['gender']
        venue = match_info['venue']

        match_stats = {
            'team1': team1,
            'team2': team2,
            'toss_winner': toss_winner,
            'winner': winner,
            'gender': gender,
            'venue': venue
        }

        return match_stats

file_paths = [os.path.join(extract_path, file) for file in os.listdir(extract_path) if file.endswith('.json')]

all_matches_data = []
for file_path in file_paths:
    match_data = load_json(file_path)
    if match_data:
        all_matches_data.append(match_data)

df = pd.DataFrame(all_matches_data)

df.dropna(inplace=True)

# Encode categorical variables
df_encoded = pd.get_dummies(df, columns=['team1', 'team2', 'toss_winner', 'gender', 'venue'])

# Prepare data for the model
features = df_encoded.drop('winner', axis=1)
target = df['winner']

label_encoder = LabelEncoder()
target_encoded = label_encoder.fit_transform(target)

X_train, X_test, y_train, y_test = train_test_split(features, target_encoded, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Serialize the model and label encoder
joblib.dump(model, 'random_forest_model.joblib')
joblib.dump(label_encoder, 'label_encoder.joblib')

# Save preprocessed data in JSONL format
df.to_json('combined_matches.jsonl', orient='records', lines=True)

print("Model, label encoder, and combined matches data saved successfully.")
