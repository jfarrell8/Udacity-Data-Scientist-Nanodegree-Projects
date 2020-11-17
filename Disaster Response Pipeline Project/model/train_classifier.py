import sys
import pandas as pd
import numpy as np
import re
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sqlalchemy import create_engine
from nltk import word_tokenize, sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import pickle
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

def load_data(database_filepath):
    """
    Load data up from the database_filepath created in '../data/process_data.py'
    
    Args:
        database_filepath: the filepath from which to grab the database needed to load the data
        
    Returns:    
        X (df): X training data from etl_disaster_pipeline table in the db
        y (df): y training data from etl_disaster_pipeline table in the db
        category_names (Series): columns (category) names from the etl_disaster_pipeline table in the db
    """
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('etl_disaster_pipeline', con=engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = Y.columns

    return X, Y, category_names

def tokenize(text):
    """
    Function that tokenizes text.
    
    Args:
        text (string): text to tokenize
    
    Returns:
        clean_tokens (list): tokenized version of text
    """
    # remove special characters and lowercase
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Build a ML model and tune the hyperparameters of the model by creating a pipeline,
    establishing the parameters of interest, and running GridSearchCV
    on the model/hyperparameters to identify the "optimal" model.
    
    Returns:
        cv: The best model
    """
    # create pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(DecisionTreeClassifier()))
        ])

    # create parameter bounds for hyperparameter tuning
    parameters = {
        #'clf__estimator__n_neighbors': [5, 8] (for when I ran k nearest neighbors)
        'clf__estimator__max_depth': [2, 3]
    }

    # instantiate a GridSearch object to find better hyperparameters
    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=6, verbose=2)

    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Use the model to evaluate test data and compare predictions to actual test outputs.
    Print out classification reports for each category in Y_test data.
    
    Args:
        model: ML model to predict from
        X_test (df): the testing data for our features
        Y_test (df): labeled testing data
        category_names (Series): list of category names
    """
    
    # predict categories from the feature test data
    y_pred = model.predict(X_test)
    y_pred = pd.DataFrame(y_pred, columns=category_names)

    # print out classification report for each category
    for i, cat in enumerate(y_pred.columns):
        print(cat)
        print(classification_report(Y_test.iloc[:, i], y_pred.iloc[:, i]))


def save_model(model, model_filepath):
    """
    Saves our model to a pickle file via a provided model filepath.
    
    Args:
        model: ML model to save
        model_filepath: path destination for the pickle file
    """
    
    saved_model = pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()