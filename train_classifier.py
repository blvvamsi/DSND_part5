# Importing all the libraries required for this project
import re
import sys
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score, precision_score
from sklearn.metrics import recall_score , f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import warnings

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


def load_data(database_pth, table_name='CleanedDataTable'):
    """Loading the cleaned data from the database into a df.

    Arguments:
        database_pth:Is a String ,  it contains the  cleaned data table saved during ETL Pipeline.
        table_name: Is a String, it contains the cleaned data table.

    Returns:
       X: numpy array consisting of the disaster messages.
       Y: numpy array consisting of the disaster category for each message.
       cat_names: list. Disaster category names.
    """
    # loading the data from database
    engine = create_engine('sqlite:///' + database_pth)

    # reading the new table we created in the ETL pipeline preparation
    df = pd.read_sql_table(table_name, con=engine)

    cat_names = df.columns[4:]

    X = df[['message']].values[:, 0]
    y = df[cat_names].values

    return X, y, cat_names




def tokenize_fn(text, lemmatizer=WordNetLemmatizer()):
    
    url_re = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    det_urls = re.findall(url_re, text)
    for url in det_urls:
        text = text.replace(url, 'urlplaceholder')
    
    # tokenizing
    tokens = nltk.word_tokenize(re.sub(r"[^a-zA-Z0-9]", " ", text.lower()))
    
    # Stopword removal
    tokens = [t for t in tokens if t not in stopwords.words('english')]

    # lemmatizing
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    
    return tokens


def model_stats(X, Y, cat_names, vocab_pth, cat_pth):
    """Save stats

    Args;
        X: numpy array which consists of the Disaster messages.
        Y: numpy array which consists of the Disaster categories for each message.
        cat_names: Disaster category names.
        vocab_pth:Is a String,  Vocabulary stats are saved as a pickle into this file.
        cat_pth:Is a String, Category stats are saved as a pickle into this file.
    """
    # Check vocabulary
    vect = CountVectorizer(tokenizer=tokenize_fn)
    X_vectorized = vect.fit_transform(X)

    # Convert vocabulary into pandas.dataframe
    keys, values = [], []
    for k, v in vect.vocabulary_.items():
        keys.append(k)
        values.append(v)
    vocab_df = pd.DataFrame.from_dict({'words': keys, 'counts': values})

    # Vocabulary stats
    vocab_df = vocab_df.sample(30, random_state=72).sort_values('counts', ascending=False)
    vocab_counts = list(vocab_df['counts'])
    vocab_words = list(vocab_df['words'])

    # Save vocabulary stats
    with open(vocab_pth, 'wb') as vocab_file:
        pickle.dump((vocab_counts, vocab_words), vocab_file)

    # Category stats
    cat_counts = list(Y.sum(axis=0))

    # Save category stats
    with open(cat_pth, 'wb') as cat_stats_file:
        pickle.dump((cat_counts, list(cat_names)), cat_stats_file)



def build_model():
    """Building the model.

    Returns:
        pipeline
    """
    
    
    pipeline_imp = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize_fn)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(
            AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1, class_weight='balanced'))
        ))
    ])

    
    # Improved parameters 
    parameters_imp = {
        'clf__estimator__n_estimators': [100, 200],
        'clf__estimator__learning_rate': [0.1, 0.3]
    }

    # new model with improved parameters
    cv_imp = GridSearchCV(estimator=pipeline_imp, param_grid=parameters_imp, cv=3, scoring='f1_weighted', verbose=3)

    return cv_imp


def evaluate_model(model, X_test, y_test, cat_names):
    """Evaluate model

    Args:
        model: sklearn.model_selection.GridSearchCV.  It contains a sklearn estimator.
        X_test: numpy array consisting of the Disaster messages.
        y_test: numpy array consisting of Disaster categories for each messages
        cat_names: Disaster category names.
    """
    # Predicting the  categories of messages based on our model.
    y_pre = model.predict(X_test)

    # Print the metrics of the model
    for p in range(0, len(cat_names)):
        print(cat_names[p])
        print("\tAccuracy: {:.4f}\t\t% Recall is: {:.4f}% Precision score is: {:.4f}% F1_score is : {:.4f}".format(
            accuracy_score(y_test[:, p], y_pre[:, p]),
            recall_score(y_test[:, p], y_pre[:, p], average='weighted'),
            precision_score(y_test[:, p], y_pre[:, p], average='weighted'),
            f1_score(y_test[:, p], y_pre[:, p], average='weighted')
        ))


def save_model(model, model_pth):
    """Save model

    Args:
        model: sklearn.model_selection.GridSearchCV. It contains a sklearn estimator.
        model_pth: saved as a pickle into this file
    """
    with open(model_pth, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 5:
        database_pth, model_pth , vocab_pth, cat_pth = sys.argv[1:]
        print('Loading the data.....\n    DATABASE is : {}'.format(database_pth))
        X, Y, cat_names = load_data(database_pth)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Saving stats...')
        model_stats(X, Y, cat_names, vocab_pth, cat_pth)

        print('Building model...')
        model = build_model()

        print('Training model...')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, cat_names)

        print('Saving model...\n    MODEL: {}'.format(model_pth))
        save_model(model, model_pth)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl vocabulary_stats.pkl category_stats_pkl')


if __name__ == '__main__':
    main()
