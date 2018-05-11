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


def evaluate_Models(x,y):
    #Evaluate different machine learning algorithms on dataset using cross validation.

    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('RFC', RandomForestClassifier()))
    models.append(('G NB', GaussianNB()))
    models.append(('Mul NB', MultinomialNB()))

    # evaluate each model in turn
    seed = 10 # Random state seed to use
    results = [] #Define lists that are going to store results
    names = []
    scoring = 'accuracy' #Use accuracy for scoring method
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=seed) #10 bins
        cv_results = model_selection.cross_val_score(model, x, y, #Models and data to use
                                                     cv=kfold,
                                                     scoring=scoring)
        #Append results and names to list to print out in the next line
        results.append(cv_results)
        names.append(name)
        print(name, cv_results.mean(), cv_results.std()) # Print out results from each algorithm

    # draw boxplot comparing the results from the different models
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()

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

def main():
    #Load CSV file and clean the text
    df = text_Cleaner("lang_data.csv")

    #Up sample the undersampled "Nederlands" Class
    df2 = re_sample(df)
    #Split data into test and train data
    X_train, X_test, y_train, y_test = train_test_split(df2['text'], df2['language'],
                                                        test_size=0.2, #Use 80% of the data for training
                                                        random_state = 123)
    # count_vect = CountVectorizer()
    # X_train_counts = count_vect.fit_transform(X_train) # Calculate mean and standard deviation and Convert strings to
                                                           # vectors as all models require numerical input and centre data
    # evaluate_Models(X_train_counts,y_train)                    # This function was used to evaluate different classification algorithms
    # X_test_counts = count_vect.transform(X_test)                 # Centre data and convert strings of testing data x

    # Apply the multinomial naive bayes algorith to the training data
    pipe = make_pipeline(CountVectorizer(), MultinomialNB())
    pipe.fit(X_train, y_train)
    # clf = MultinomialNB().fit(X_train_counts, y_train)
    # Save predictions obtained
    y_pred = pipe.predict(X_test)
    # Do data analysis in form of classification report and confusion matrix
    class_report = classification_report(y_test, y_pred)
    Con_matrix = pd.crosstab(y_test, y_pred)
    # Con_matrix = confusion_matrix(y_test, y_pred,labels=['English','Afrikaans','Nederlands'])

    print ('Classification report: {},Confusion Matrix {}'.format(class_report,Con_matrix))
    print('Score: {}'.format(pipe.score(X_test, y_test)))
    # print(clf.predict(count_vect.transform(["dat gaat dus weer als vanzelf"])))

    # Save the trained program to a file that can be used later on for applying the model
    joblib.dump(pipe, 'model.pkl')
main()


