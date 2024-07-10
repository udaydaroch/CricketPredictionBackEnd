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
