# Project Title : Disaster Response Project

This project is part of the Udacity Data Scientist Nanodegree. The goal is to build a machine learning pipeline to categorize disaster messages so that appropriate disaster response teams can be notified.

## Project Structure

### 1. ETL Pipeline
- **ETL Notebook**: Contains detailed code for the ETL (Extract, Transform, Load) process.
- **process_data.py**: 
  - Extracts data from two datasets.
  - Merges the data into one.
  - Cleans and stores the data in a SQLite database.
  - Detailed code is defined in the ETL Notebook.

### 2. ML Pipeline
- **ML Notebook**: Contains detailed code for machine learning tasks.
- **train_classifier.py**: 
  - Loads data from the SQLite database.
  - Splits the data into training and test sets.
  - Builds a machine learning pipeline.
  - Trains and tunes a model using GridSearchCV.
  - Evaluates the model.
  - Saves the trained model as a pickle file.
  - Detailed code is defined in the ML Notebook.

### 3. Project Database
- **DisasterResponse.db**: 
  - Created from the `process_data.py` script to store cleaned data.

### 4. Project Datasets
- **Categories.csv**: Contains the categories for each message.
- **Messages.csv**: Contains the disaster messages.

## Getting Started

### Prerequisites
- Python 3.x
- Required libraries: pandas, numpy, sqlalchemy, scikit-learn, nltk, pickle, joblib

### Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run the ETL pipeline that cleans data and stores it in the database:
      ```sh
      python process_data.py data/messages.csv data/categories.csv data/DisasterResponse.db
      ```

    - To run the ML pipeline that trains the classifier and saves the model:
      ```sh
      python train_classifier.py data/DisasterResponse.db models/classifier.pkl
      ```

2. You can find detailed code and explanations in the ETL and ML Notebooks.

## Acknowledgements
This project is part of the Udacity Data Scientist Nanodegree program.
