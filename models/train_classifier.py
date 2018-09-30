#! /usr/bin/env python3
# coding=utf-8

# The model training script

import sys
import re
import nltk

import pandas as pd
import pickle
from sqlalchemy import create_engine

# import statements
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def load_data(database_filepath):
    """Load  a database table and return
    values, labels and category names

    Parameters
    ----------
    database_filepath : string
        location of the database

    Returns
    -------
    X: numpy.ndarray
        The training data
    Y: numpy.ndarray
        The training labels
    category_names: list
     The labely names category names
    """

    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('messages_categories', engine)

    X = df.message.values
    Y = df.iloc[:, 4:].values

    category_names = (df.columns[4:]).tolist()

    return X, Y, category_names


def tokenize(text):
    """Tokenize text

    Parameters
    ----------
    text : string
        the tect to tokenize

    Returns
    -------
    clean_tokens : list
        the tokens list

    """

    # normalize
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)

    # tokenize
    tokens = word_tokenize(text)

    # remove stop words
    words = [w for w in tokens if w not in stopwords.words("english")]

    # initiate Lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Lemmatize, convert to lower case and strip
    clean_tokens = []
    for word in words:
        clean_word = lemmatizer.lemmatize(word).lower().strip()
        clean_tokens.append(clean_word)

    return clean_tokens


def build_model():
    """Build and optimize model

    Parameters
    ----------
    None

    Returns
    -------
    None

    """

    # build pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(), n_jobs=1)),
    ])

    # set parameters for grid search
    parameters = {
        'clf__estimator__criterion': ['gini', 'entropy'],
        'tfidf__use_idf': [True, False],
        'tfidf__norm': ['l1', 'l2']
    }

    # optimize model
    model = GridSearchCV(pipeline, param_grid=parameters,
                         cv=2, verbose=1)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluates and prints model performance

    Parameters
    ----------
    model : multiclassification model
    X_test: numpy.ndarray
        The test data
    Y_test: numpy.ndarray
        The test labels
    category_names: list
     The category names

    Returns
    -------
    None

    """

    Y_pred = model.predict(X_test)

    for i in range(len(category_names)):
        print("Label:", category_names[i])
        print("---- Classification Repor")
        print(classification_report(Y_test[:, i], Y_pred[:, i]))


def save_model(model, model_filepath):
    """Save model as a pickle file

    Parameters
    ----------
    model : multiclassification model
        The optimized classifier
    model_filepath : string
        location of the database

    Returns
    -------
    None

    """
    with open(model_filepath, 'wb') as outfile:
        pickle.dump(model, outfile)


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
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
