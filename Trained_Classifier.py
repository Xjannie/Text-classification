# with open('filename.pkl', 'rb') as f:
import pandas as pd
import matplotlib.pyplot as plt
import string
import pickle
import joblib
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
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

def re_sample(df):
    # Divide dataframe into different classes
    df_Eng = df[df["language"] == "English"]
    df_Afr = df[df["language"] == "Afrikaans"]
    df_Ned = df[df["language"] == "Nederlands"]

    # Upsample Nedelrands class
    df_Ned_resampled = resample(df_Ned, replace=True,   # sample with replacement
                                     n_samples=637,     # to match Afrikaans class
                                     random_state=123)  # Random number

    # Combine Nederlands class with other classes to give a database where Nedelands is not under sampled
    df_resampled = pd.concat([df_Eng, df_Afr,df_Ned_resampled])

    # Display new class counts
    print(df_resampled['language'].value_counts())
    return df_resampled
def remove_punctuation(x):
    #   Remove Punctuation
    s = ''.join([i for i in x if i not in frozenset(string.punctuation)])
    return s

def main():
    # prepare the pipeline


    # Loading the saved model with joblib
    pipe = joblib.load('model.pkl')

    # New data to predict
    df = text_Cleaner("lang_data.csv")
    X_test = df['text']
    y_test = df ['language']

    y_pred = pipe.predict(X_test)
    class_report = classification_report(y_test, y_pred)
    print(class_report)

main()