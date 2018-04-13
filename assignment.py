""""
author: Sam Luby
created: 10th april 2018
"""

import numpy as np
import urllib
from bs4 import BeautifulSoup as bs
import os
import sklearn
import pandas as pd
import scipy
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from scipy.cluster.hierarchy import ward, dendrogram
import matplotlib.pyplot as plt


###########################
# Part 1: Data Collection #
###########################
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

###############################
# Part 2: Text Classification #
###############################
# 1. From the files created in Part 1, load the set of raw documents into your
# notebook. Ensure that each document has a class label, based on the original
# category label that you identified.

directory_category = "categories/categories.txt"
directory_articles = "data/"

def load_categories(directory):
    with open(directory) as f:
        categories = f.readlines()
    return categories


def load_articles_filenames(directory):
    articles = []
    for file in os.listdir(directory):
        articles.append(file)
    return articles


# natural sorting for each article file name, which is type string
def natural_sorting(file):
    return [int(c) if c.isdigit() else c for c in re.split('(\d+)', file)]


categories = load_categories(directory_category)
articles = load_articles_filenames(directory_articles)
number_of_articles = len(articles)
articles.sort(key=natural_sorting)
df = pd.DataFrame({'CATEGORY': categories, 'ARTICLE': articles})


def load_all_articles_contentes():
    articles = list()
    for i in range(number_of_articles):
        file = open("data\\article" + str(i) + ".txt", "r")
        content = file.read()
        articles.append(content)
    return articles


# 2. From the raw documents, create a document-term matrix, using appropriate
# text pre-processing and term weighting steps.

def tokenizer(text):
    #tokenizer
    tokenize = CountVectorizer().build_tokenizer()
    tokens = tokenize(text)

    #lemmatisation
    lemmatizer = nltk.stem.WordNetLemmatizer()
    tokens_lemmatized = []
    for token in tokens:
        tokens_lemmatized.append(lemmatizer.lemmatize(token))
    return tokens_lemmatized


def create_term_matrix(articles):
    words = []
    for i in articles:
        lemma_tokens = tokenizer(i)
        words.extend(lemma_tokens)

    # create TF-IDF weighted document-term matrix
    vectorizer = TfidfVectorizer(min_df = 5, stop_words='english', tokenizer=tokenizer, ngram_range=(1,3))
    term_matrix = vectorizer.fit_transform(articles)
    return term_matrix

articles = load_all_articles_contentes()
term_matrix = create_term_matrix(articles)
distance_matrix = 1 - cosine_similarity(term_matrix)

linkage_matrix = ward(distance_matrix)
plt.figure(figsize=(15, 250))
# dendrogram(linkage_matrix, orientation="right",labels=art_index) #hierarchical clustering as a dendrogram.
# plt.show()