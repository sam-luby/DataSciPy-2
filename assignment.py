""""

author: Sam Luby
created: 10th april 2018
"""

import numpy as np
import urllib
from bs4 import BeautifulSoup as bs
import os

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
    file = open("categories.txt", "w")
    categories = ""

    # for the given monthly sub-page, put each article category into a file
    for category in monthly_contents.find_all('td', class_='category'):
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
        article_text += (paragraph.get_text() + '\n')
    return article_text


# save the contents of each article into a separate text file.
def save_article_contents(articles):

    # create directory for saving articles
    dirname = "dat"
    os.mkdir(dirname)

    # save the contents of each article into a seperate text file
    for i in range(len(articles)):
        article_text = get_article_content(articles[i])
        path = "dat/" + str(i) + "_article.txt"
        file = open(path, "w", encoding="utf-8")
        file.write(article_text)
        print('Saved article #' + str(i))
        file.close()


articles_list = get_articles_list(baseurl)
save_article_contents(articles_list)