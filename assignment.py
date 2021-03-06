""""
author: Sam Luby
data science in python assignment 2
created: 10th april 2018
"""

import numpy as np
import urllib
from bs4 import BeautifulSoup as bs
import os
import pandas as pd
import itertools
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import *

######################################################################################################
                                ############################
                                # Part 1:  Data Collection #
                                ############################
######################################################################################################
baseurl = "http://mlg.ucd.ie/modules/COMP41680/archive/"


# get a list of all articles in all sub-pages of a website
def get_articles_list(baseurl):
    articles_list = list()
    months_list = list()

    # open url and get page contents
    f = urllib.request.urlopen(baseurl + "index.html")
    dat = f.read()
    page_contents = bs(dat, 'html.parser')

    # finds the links to sub-pages (monthly sub-page) on this index page (only legitimate links)
    for subpage in page_contents.find_all('a'):
        if len(subpage.get('href')) > 0:
            months_list.append(subpage.get('href'))

    # finds the links on the monthly sub-page which contains links to each article
    for i in range(len(months_list)):
        monthly_subpage = baseurl + months_list[i]
        f = urllib.request.urlopen(monthly_subpage)
        dat = f.read()
        monthly_contents = bs(dat, 'html.parser')

        # calls the get_category_labels function to add categories for each month to a file
        get_category_labels(monthly_contents)

        # get each link (each article) for the given month, append to list.
        for subpage in monthly_contents.find_all('a'):
            url = subpage.get('href')
            if "article" not in url:
                continue
            articles_list.append(url)
    return articles_list


# get the category labels for each month, store in text file
def get_category_labels(monthly_contents):
    # create directory for saving articles
    file = open("categories/categories.txt", "a")
    categories = ""

    # for the given monthly sub-page, put each article category into a file
    for category in monthly_contents.find_all('td', class_='category'):
        if "N/A" not in category.get_text():
            categories += category.get_text() + '\n'
    file.write(categories)
    file.close()


# gets the body of each article, by extracting plain text from <p> html tags
def get_article_content(article):

    # get each article url and open
    article_url = baseurl + article
    f = urllib.request.urlopen(article_url)
    dat = f.read()
    article_contents = bs(dat, 'html.parser')

    # get the content of each article, the text in each legitimate <h2> paragraph
    article_text = article_contents.find('h2').get_text()
    for paragraph in article_contents.find_all('p', class_=''):
        article_text += (paragraph.get_text() + "\n")
    return article_text


# save the contents of each article into a separate text file.
def save_article_contents(articles):

    # save the contents of each article into a seperate text file
    for i in range(len(articles)):
        article_text = get_article_content(articles[i])
        path = "data/" + "article_" + str(i) + ".txt"
        file = open(path, "w", encoding="utf-8")
        file.write(article_text)
        if i % 10 == 0:
            print('Saved article #' + str(i))
        file.close()


# only download  files if necessary
def download_files(baseurl):
    data_path = "data"
    categories_path = "categories"

    # check if directory exists, if not then download everything
    if not os.path.isdir(data_path):
        print("Files not present, downloading")
        os.mkdir(data_path)
        os.mkdir(categories_path)
        articles_list = get_articles_list(baseurl)
        save_article_contents(articles_list)
    else:
        print("Files already present.")



download_files(baseurl) # main function for data scraping


######################################################################################################
                                ###############################
                                # Part 2: Text Classification #
                                ###############################
######################################################################################################
# Part 2.1: Load documents

directory_category = "categories/categories.txt"
directory_articles = "data/"

# load each category from file
def load_categories(directory):
    with open(directory) as f:
        categories = [line.rstrip() for line in f]
    return categories


# load all filenames from subdirectory
def load_articles_filenames(directory):
    articles = []
    for file in os.listdir(directory):
        articles.append(file)
    return articles


# natural sorting for each article file name, which is type string
def natural_sorting(file):
    return [int(c) if c.isdigit() else c for c in re.split('(\d+)', file)]


# create pandas dataframe from categories and article filenames
def create_dataframe():
    categories = load_categories(directory_category)
    articles = load_articles_filenames(directory_articles)
    articles.sort(key=natural_sorting)
    df = pd.DataFrame({'CATEGORY': categories, 'ARTICLE': articles})
    return df


# load the contents (text) from each article
def load_all_articles_contents(number_of_articles):
    articles = list()
    articles_index = list()

    # for each article, append the contents and index to seperate lists
    for i in range(number_of_articles):
        file = open("data\\article_" + str(i) + ".txt", "r", encoding='utf-8')
        content = file.read()
        articles.append(content)
        articles_index.append(int(i))
    return articles, articles_index


df = create_dataframe()
number_of_articles = len(df)


# Part2.2: Create a document-term matrix

# tokenizer and lemmatizer
def lemma_tokenizer(text):
    #tokenizer
    tokenize = CountVectorizer().build_tokenizer()
    tokens = tokenize(text)

    #lemmatisation
    lemmatizer = nltk.stem.WordNetLemmatizer()
    tokens_lemmatized = []
    for token in tokens:
        tokens_lemmatized.append(lemmatizer.lemmatize(token))
    return tokens_lemmatized


# document term-matrix (frequency of terms)
def create_term_matrix(articles):
    words = []
    for i in articles:
        lemma_tokens = lemma_tokenizer(i)
        words.extend(lemma_tokens)

    # create TF-IDF weighted document-term matrix
    vectorizer = TfidfVectorizer(min_df=3, stop_words='english', tokenizer=lemma_tokenizer, ngram_range=(1, 3))
    term_matrix = vectorizer.fit_transform(articles)
    return term_matrix, vectorizer, words


# print the list of the N most common words found in articles
def get_N_most_common_words(N, vectorizer, words):
    most_common = list()
    index = np.argsort(vectorizer.idf_)[::-1]
    for i in range(0, N):
        most_common.append(words[index[i]])
    print(most_common)


articles, articles_index = load_all_articles_contents(number_of_articles)
term_matrix, vectorizer, words = create_term_matrix(articles)
# print(term_matrix)
# print(words)
# print(vectorizer)
# get_N_most_common_words(10, vectorizer, words)


####################################
#       Classification Models      #
####################################

# gets the article categories (no duplicates)
def get_categories(df):
    categories = df[['CATEGORY']].drop_duplicates()['CATEGORY'].tolist()
    for i in  range(len(categories)):
        categories[i] = categories[i].replace(u'\xa0', '')
    return categories

categories = get_categories(df)
print(categories)

# isolate category column from dataframe
target = df['CATEGORY']
data_train, data_test, target_train, target_test = train_test_split(term_matrix, target, test_size=0.2)

# kNN - Nearest Neighbour
model_kn = KNeighborsClassifier(n_neighbors=3)
model_kn.fit(data_train, target_train)
predicted_kn = model_kn.predict(data_test)
knn_acc = accuracy_score(target_test, predicted_kn)

# Naive Bayes
model_nb = MultinomialNB()
model_nb.fit(data_train, target_train)
predicted_nb = model_nb.predict(data_test)
nb_acc = accuracy_score(target_test, predicted_nb)


# Support Vector Machines
model_svc = SVC(kernel='linear')
model_svc.fit(data_train, target_train)
predicted_svc = model_svc.predict(data_test)
svc_acc = accuracy_score(target_test, predicted_svc)


####################################
# Evaluating Classification Models #
####################################

# Accuracy
print("Accuracy using kNN method: " + str(round(knn_acc*100, 2)) + "%")
print("Accuracy using Naive Bayes method: " + str(round(nb_acc*100, 2)) + "%")
print("Accuracy using SVM method: " + str(round(svc_acc*100, 2)) + "%")


# Cross-Validation Evaluation
scores = cross_val_score(model_kn, term_matrix, target, cv=5, scoring='accuracy')
print("Evaluating kNN classifier with Cross-Validation yields avg score: " + str(round(scores.mean()*100, 2)) + "%")
scores = cross_val_score(model_nb, term_matrix, target, cv=5, scoring='accuracy')
print("Evaluating Naive Bayes classifier with Cross-Validation yields avg score: " + str(round(scores.mean()*100, 2)) + "%")
scores = cross_val_score(model_svc, term_matrix, target, cv=5, scoring='accuracy')
print("Evaluating SVC classifier with Cross-Validation yields avg score: " + str(round(scores.mean()*100, 2)) + "%")


# Confusion Matrix Evaluation
cm_knn = confusion_matrix(target_test, predicted_kn)
print("kNN Confusion Matrix" + '\n' +  str(cm_knn))
cm_nb = confusion_matrix(target_test, predicted_nb)
print("Naive Bayes Confusion Matrix" + '\n' +  str(cm_nb))
cm_svc = confusion_matrix(target_test, predicted_svc)
print("SVM Confusion Matrix" + '\n' +  str(cm_svc))


# graph of confusion matrix
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True category')
    plt.xlabel('Predicted category')



# plot individual confusion matrices
plt.figure()
plot_confusion_matrix(cm_knn, categories, title='kNN Confusion Matrix')
plt.figure()
plot_confusion_matrix(cm_nb, categories, title='Naive Bayes Confusion Matrix')
plt.figure()
plot_confusion_matrix(cm_svc, categories, title='SVM Confusion Matrix')
plt.show()