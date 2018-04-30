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
import scipy
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from scipy.cluster.hierarchy import ward, dendrogram
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
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
        print('Saved article #' + str(i))
        file.close()

# os.mkdir("data")
# os.mkdir("categories")
# articles_list = get_articles_list(baseurl)
# save_article_contents(articles_list)


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


# Part 2.3: Multi-Class Classification Models
target = df['CATEGORY']
data_train, data_test, target_train, target_test = train_test_split(term_matrix, target, test_size=0.2)

# kNN - Nearest Neighbour
model = KNeighborsClassifier(n_neighbors=3)
model.fit(data_train, target_train)
predicted = model.predict(data_test)
knn_acc = accuracy_score(target_test, predicted)

# Naive Bayes
# Uses Bayes' theorem with the "naive" assumption of independence between every pair of features.
model = MultinomialNB()
model.fit(data_train, target_train)
predicted = model.predict(data_test)
nb_acc = accuracy_score(target_test, predicted)


# Support Vector Machines
model = SVC()
model.fit(data_train, target_train)
predicted = model.predict(data_test)
svc_acc = accuracy_score(target_test, predicted)

####################################
# Evaluating Classification Models #
####################################

# Accuracy
print("Accuracy using kNN method: " + str(round(knn_acc*100, 2)) + "%")
print("Accuracy using Naive Bayes method: " + str(round(nb_acc*100, 2)) + "%")
print("Accuracy using SVM method: " + str(round(svc_acc*100, 2)) + "%")


# Cross-Validation Evaluation
model = KNeighborsClassifier(n_neighbors=3)
scores = cross_val_score(model, data_train, target_train, cv=5, scoring ='accuracy')
print("Evaluating kNN classifier with Cross-Validation yields avg score: " + str(scores.mean()))

model = MultinomialNB()
scores = cross_val_score(model, data_train, target_train, cv=5, scoring="accuracy")
print("Evaluating Naive Bayes classifier with Cross-Validation yields avg score: " + str(scores.mean()))

model = SVC()
scores = cross_val_score(model, data_train, target_train, cv=5, scoring="accuracy")
print("Evaluating SVC classifier with Cross-Validation yields avg score: " + str(scores.mean()))


# Confusion Matrix Evaluation
model = KNeighborsClassifier(n_neighbors=3)
model.fit(data_train, target_train)
predicted = model.predict(data_test)
cm = confusion_matrix(target_test, predicted)
print(cm)