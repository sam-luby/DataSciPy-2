# DataSciPy-2

Data Science in Python - Assignment 2 (A+ grade achieved)
project structure:
1. assignment2.ipynb - The Jupyter notebook for nice step-by-step procedure including graphs and discussion of results.
2. assignment2.py - The source code used to create the notebook.
3. data - folder with all scraped web content

The project was an introduction to web scraping, machine learning and text processing.


This assignment can be broken down into the following:
1. Collecting and filtering a large accumulation of content from a given website (web scraping).
2. Text processing and text mining to create a document-term matrix - a weighted list of the base-form of terms that appear    in the text.
3. Build 3 different text classification models: kNN, Naive Bayes & SVM. Apply them to the text data. 
4. Evaluate the performance of the text classification models based on different evaluation parameters. 


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


The corpus and categories are then each split into two seperate sets - training sets and test sets (to train and then evaluate each model).
* _data_train_ is the data to train the model.
* _data_test is_ the data used for evaluation.
* _target_train_ is the category labels to train the model.
* _target_test_ is the category labels used for evaluation.

The ratio of 1:4 was used to split the sets - i.e. 20% of the data was used to train the model to test the remaining 80%.


I then applied the sets to each of the three classification models: first training the model with _data_train_ and _data_test_. Then, predictions for the test data are made for each of the classification models, which will be used for evaluation purposes.


### Evaluation of Classification Models
Then the classification models are evaluated on their performance, based on various performance characteristics.

#### Accuracy 
The simplest characteristic. It is the fraction or percentage of correct predictions to total predictions made by the classifier.

#### Cross-Validation Evaluation
Cross-validation involves parititioning the original data into distinct subsamples or 'folds', where each fold contains the same proportion of the corpus. The experiment is repeated for all folds and the accuracy is an average of each run.

I experimented on several fold values, where in theory the higher number of folds yields the most accurate evaluation (n-folds gives n-values which are averaged).

#### Confusion Matrix Evaluation
The final method of evaluation I used is a confusion or error matrix. This is a way of visualising the model performance with a table layout. I used pyplot to create a more aesthetic table. 

<image> 
  
The table shows the proportion of correct predictions.
The above three plots show the performance of classifiers when evaluating using the confusion matrix. The values are normalised, reporting a ratiometric value of correct predictions to all predictions.  

Click into the jupyter notebok (.ipynb file) for a more detailed step-by-step procedure.
