import numpy as np
import urllib
from bs4 import BeautifulSoup as bs

###########################
# Part 1: Data Collection #
###########################

# get a list of all articles in all subpages of a website
def get_articles_list(page_contents):
    articles_list = list()
    months_list = list()
    for subpage in page_contents.find_all('a'):
        if len(subpage.get('href')) > 0:
            months_list.append(subpage.get('href'))
    for i in range(len(months_list)):
        monthly_subpage = baseurl + months_list[i]
        f = urllib.request.urlopen(monthly_subpage)
        dat = f.read()
        monthly_contents = bs(dat, 'html.parser')
        for subpage in monthly_contents.find_all('a'):
            url = subpage.get('href')
            if "article" not in url:
                continue
            articles_list.append(url)

    print(months_list)
    for i in range(len(articles_list)):
        print(articles_list[i])


baseurl = "http://mlg.ucd.ie/modules/COMP41680/archive/"
f = urllib.request.urlopen(baseurl + "index.html")
dat = f.read()
page_contents = bs(dat, 'html.parser')
print(page_contents.prettify())
get_articles_list(page_contents)


