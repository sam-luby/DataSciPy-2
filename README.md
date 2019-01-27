# DataSciPy-2

Data Science in Python assignment 2

Assignment involved web scraping, text processing, applying text classification models and evaluating their performance. 

We were provided with a URL (now gone but all data is saved here) with 1408 various news articles.

BeautifulSoup is used to parse the articles and filter dead/unwanted webpage links.
The corpus (article text contents) for each article are saved in seperate text files, along with the category the article belongs to (tech, business or sport).


The articles are loaded into a pandas dataframe and some text processing is performed:

-> the text is lemmatized: words are grouped together in terms of their based form (eg walking/walked -> walk)

-> the text is tokenized: the text is divided into a series of tokens.

-> a document-term matrix is created using the above: this is the frequency of words after lemmatizing/tokenizing.

-> the text is vetorized: exluding stop-words and removing very infrequency ( < 5 ) terms. 

-> finally a term frequency-inverse document frequency (TF-IDF) maxtris is created. this involves weighting terms based on their importance (ie 'as', 'of', 'the' etc are 'weighed-down')


Three different classification models are used on the text:
k-Nearest Neighbours Classifier, the Naive Bayes classifier and the Support Vector Machines model.

The text corpus is split into training and test sets (training to train the model, test to evaluate the model).
The same is done for the list of categories.

The 3 text classification models are used and evaluated on their performance.
Various performance characteristics are used to evaluate the models.

Click into the jupyter notebok (.ipynb file) for the step-by-step procedure.

