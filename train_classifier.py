import sys
import nltk
import re
nltk.download(['punkt', 'wordnet'])
import warnings
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import joblib
from sklearn.metrics import classification_report, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

def load_data(database_filepath):
    '''loading the data from the database.'''
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table('Messages', engine)
    X = df.message
    Y = df[df.columns[4:]]
    category_names = Y.columns
    return X, Y, category_names

def tokenize(text):
    ''' Tokenizing text.'''
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    '''creating pipeline'''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize,token_pattern=None)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(OneVsRestClassifier(LinearSVC(random_state = 0))))
    ])
    parameters = {
                'tfidf__smooth_idf':[True, False],
                'clf__estimator__estimator__C': [1, 2, 5]
             }
    cv = GridSearchCV(pipeline, param_grid=parameters, scoring='precision_samples', cv = 5)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''Evaluating model'''
    Y_pred = model.predict(X_test)
    for i in range(len(category_names)):
        print(f"Category: {category_names[i]}")
        print(classification_report(Y_test.iloc[:, i], Y_pred[:, i]))
        print("\n")


def save_model(model, model_filepath):
    '''Saving model.'''
    joblib.dump(model, model_filepath)


def main():
    
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        model = build_model()
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('missing paths!!')


if __name__ == '__main__':
    main()
