# Cricket Match Winner Prediction

This project includes scripts to train a machine learning model to predict the winner of T20 cricket matches and serve the predictions via a Flask API.

## Overview

The project consists of two main components:
1. **prepare_model.py**: A script to prepare and train a machine learning model.
2. **app.py**: A script to run a Flask API that serves the predictions from the trained model.

## Data Source

The data is sourced from [cricsheet.org](https://cricsheet.org), specifically the T20 match data in JSON format.

## Features

- **Model Training**: Trains a Random Forest classifier on match data.
- **API**: Provides endpoints to retrieve match options and predict match outcomes.

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)
- Virtual environment (recommended)

### Steps

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
2. **Create and activate a virtual environment**
    ```bash
    python -m venu venu
    source ven/bin/activate # On windows use `venu\Scripts\activate
3. **Install dependencies**
   ```bash
   cors, Flask, joblib, panda

## Usage

### Model Training 
- Run the `prepare_model.py` script to download the data, prepare it, and train the model: 
   ```bash
   python prepare_model.py

### The script will perform the following steps:

## **Data Collection**
- Downloads a zip file containing T20 match data in JSON format from cricsheet.org.
## **Data Extraction**
- Extracts JSON files from the downloaded zip file.
- Reads each JSON file to extract relevant match-level data such as teams, toss winner, match
  winner, gender, and venue.
## **Data Preparation**
- Collects data from all JSON files into a pandas DataFrame.
- Removes rows with missing data.
- Encodes categorical variables using one-hot encoding.
## **Model Training**
- Prepares features and target variables, with the target being the match winner.
- Encodes the target variable using LabelEncoder.
- Splits the data into training and testing sets.
- Trains a RandomForestClassifier on the training data.
## **Model Saving**
- Saves the trained model and the label encoder using joblib.
- Saves the combined match data in JSONL format.

### Running the Flask API 

- Run the `app.py` script to start the Flask API: 
  ```bash
  python app.py
- The API will be available at http://localhost:5000.

###Flask API Endpoints

`Get /matches`
- Retrieves avaible teams and venues for matches
    ``` json
   {
     "teams": ["Team1", "Team2", ...],
     "venues": ["Venue1", "Venue2", ...]
   }
`POST /predict`

- Predicts the winner of a match based on input data
     - Request Body:
     ``` json
     {
        "team1": "Team1",
        "team2": "Team2",
        "toss_winner": "Team1",
        "gender": "male",
        "venue": "Venue1"
      }
   ```
   - Response Body:
    ``` json
   {
     "predicted_winner": "Team1",
     "predicted_probability": 75.0
    }
