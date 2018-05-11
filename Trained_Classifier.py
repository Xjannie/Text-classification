# with open('filename.pkl', 'rb') as f:
import pandas as pd
import matplotlib.pyplot as plt
import string
import pickle
import joblib
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import resample
from sklearn.pipeline import make_pipeline

filename = "lang_data.csv"

def text_Cleaner(Filename):
    df = pd.read_csv("lang_data.csv")
    # Drop empty rows in text column
    df = df[~df['text'].isnull()]

    # Drop duplicates in text column
    df = df.drop_duplicates(subset=['text'], keep=False)

    #Convert text to all lower case, as a capitol letter has no effect on which language it
    # is and will unnessaceraliy complicate the data
    df['text'] = df['text'].str.lower()
    df['text'] = df['text'].apply(remove_punctuation)
    return df


def remove_punctuation(x):
    #   Remove Punctuation
    s = ''.join([i for i in x if i not in frozenset(string.punctuation)])
    return s

def main():
    # prepare the pipeline


    # Loading the saved model with joblib
    pipe = joblib.load('model.pkl')

    # New data to predict
    df = text_Cleaner(filename)
    X_test = df['text']
    y_test = df ['language']

    y_pred = pipe.predict(X_test)
    class_report = classification_report(y_test, y_pred)
    print(class_report)
    print('Accuracy Score: {}'.format(accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)))

main()