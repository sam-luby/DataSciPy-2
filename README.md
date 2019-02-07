# DataSciPy-2

Data Science in Python - Assignment 2 (A+ grade achieved)
project structure:
1. assignment2.ipynb - The Jupyter notebook for nice step-by-step procedure including graphs and discussion of results.
2. assignment2.py - The source code used to create the notebook.
3. data - folder with all scraped web content

The project was an introduction to web scraping, machine learning and text processing.


This assignment can be broken down into the following:
i. Collecting and filtering a large accumulation of content from a given website (web scraping).
ii. Text processing and text mining to create a document-term matrix - a weighted list of the base-form of terms that appear in the text.
iii. Build 3 different text classification models: kNN, Naive Bayes & SVM. Apply them to the text data. 
iv. Evaluate the performance of the text classification models based on different evaluation parameters. 


## Procedure

### Data Collection
A link to a mock webpage was provided (it's gone now but the data is all saved in the /data/ folder).
The webpage index contained links to "montly" subpages, where each monthly subpage it self had dozens of links to various articles. Each article was given a category label on the page - sport, business or technology. The process for collecting the data from each article was as follows:
* Automatically navigate through the subpages of the index, filtering out dead links.
* On each subpage, checking each link was a valid link with article contents.
* Opening each article link and parsing the html with BeatifulSoup.
* The title and text body of the article are copied by looking at the appropriate HTML tags ('h' for headings, 'p' for paragraphs').
* The content for each article is downloaded and saved into seperate text files - 'article_0', 'article_1' etc. A running indicator shows the progress of the webscraping.
* The category labels for each article are saved into a seperate text file.

### Text Processing
I build a pandas dataframe consisting of the list of categories and the list of filenames. 'Natural sorting' must be used for the filenames, as otherwise the files would be sorted incorrectly (would be sorted as article_0, article_1, article_10, article_100 etc). 

A list of the contents of each article is also created. The contents of each is first 'tokenized', dividing the text into a series of tokens. The contents is also 'lemmatized', grouping together in terms of their based form (eg walking/walked -> walk).

A document-term matrix is created using the TF-IDF vectorizer. The TF-IDF (term frequency-inverse document frequency) weights the words to reflect their importance, so words like 'as' 'of' and 'the' are weighed down in the document-term matrix. 

### Machine Learning Classification Models
I used 3 different machine learning classification algorithms for this project.
* k Nearest Neighbours (kNN) examines 'k' nearest neighbours' labels and uses the majority vote to determine its own.
* Naive Bayes (NB) is a linear classifier based on Baye's theorem, but which assumes independence between features.
* Support Vector Machines (SVM) works by finding the hyperplane (simply, a line) that maximises the margin between two classes of data points in an N-dimensional space.




Three different classification models are used on the text:
k-Nearest Neighbours Classifier, the Naive Bayes classifier and the Support Vector Machines model.

The text corpus is split into training and test sets (training to train the model, test to evaluate the model).
The same is done for the list of categories.

The 3 text classification models are used and evaluated on their performance.
Various performance characteristics are used to evaluate the models.

Click into the jupyter notebok (.ipynb file) for the step-by-step procedure.

