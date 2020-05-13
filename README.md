# DSND_part5
This project is a part of the Udacity Data Scientist NanoDegree Program

The Packages(python3) required for running this are
Numpy , Pandas , sklearn , nltk , sqlalchemy 

MOTIVATION: The motivation behind this project is to create a Data Science pipeline and an app which classifies disaster messages into different categories. The project contains a web app where someone can input a message and get classification results in to different categories

USAGE:

First navigate to the workspace folder and run this command : python3 data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

Navigate to the workspace folder and run this command : python3 models/train_classifier.py  data/DisasterResponse.db models/classifier.pkl models/vocabulary_stats.pkl models/category_stats.pkl

This will create the pickle models to be used in our web app

Finally , navigate to thee app folder and run this command: python3 run.py

Open a new tab in your browser and go to http://127.0.0.1:3001/ where you can find the web app. It has visualizations , a panel to enter and categorise your messages

