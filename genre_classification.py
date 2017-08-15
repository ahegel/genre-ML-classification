# -*- coding: utf-8 -*-
"""
This program determines how accurately a machine learning algorithm can
classify book reviews based on the genre of the book under review.
"""

import re
import string
import yaml
from time import time
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer, TweetTokenizer

import matplotlib.pyplot as plt

import sqlite3
import os
import pandas as pd
import numpy
import operator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import GridSearchCV
from nltk import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
import pprint
from sklearn.linear_model import LogisticRegressionCV
from bs4 import BeautifulSoup
from datetime import datetime


#class TextCleaner(BaseEstimator, TransformerMixin):
#    _numeric = re.compile('(\$)?\d+([\.,]\d+)*')# with decimals and so on figures
#    _links = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')# links url
#    _trans_table = string.maketrans({key: None for key in string.punctuation})
##    def __init__(self):
##        self.name='TextCleaner'
#
#    def clean_text(self, tokens):
#        for i, w in enumerate(tokens):
#            if TextCleaner._numeric.match(w):
#                tokens[i] = w.translate(TextCleaner._trans_table)
#            elif TextCleaner._links.match(w):
#                tokens[i] = '_LINK_'
#            else:
#                continue
#        return tokens
# 
#    def transform(self, X):
#        for i, item in enumerate(X):
#            tokenized = item.split()
#            X[i] = " ".join(self.clean_text(tokenized))
#
#        return X
#
#    def fit_transform(self, X, y=None):
#        return self.transform(X)
#
#    def fit(self, X, y=None):
#        self.transform(X)
#        return self

#class EstimatorSelectionHelper:
#    '''
#    http://www.codiply.com/blog/hyperparameter-grid-search-across-multiple-models-in-scikit-learn/
#    '''
#    def __init__(self, models, params):
#        if not set(models.keys()).issubset(set(params.keys())):
#            missing_params = list(set(models.keys()) - set(params.keys()))
#            raise ValueError("Some estimators are missing parameters: %s" % missing_params)
#        self.models = models
#        self.params = params
#        self.keys = models.keys()
#        self.grid_searches = {}
#
#    def fit(self, X, y, cv=3, n_jobs=1, verbose=1, scoring=None, refit=False):
#        for key in self.keys:
#            print("Running GridSearchCV for %s." % key)
#            model = self.models[key]
#            params = self.params[key]
#            gs = GridSearchCV(model, params, cv=cv, n_jobs=n_jobs, 
#                              verbose=verbose, scoring=scoring, refit=refit)
#            gs.fit(X,y)
#            self.grid_searches[key] = gs    
#
#    def score_summary(self, sort_by='mean_score'):
#        def row(key, scores, params):
#            d = {
#                 'estimator': key,
#                 'min_score': min(scores),
#                 'max_score': max(scores),
#                 'mean_score': numpy.mean(scores),
#                 'std_score': numpy.std(scores),
#            }
#            return pd.Series(dict(params.items() + d.items()))
#
#        rows = [row(k, gsc.cv_validation_scores, gsc.parameters) 
#                     for k in self.keys
#                     for gsc in self.grid_searches[k].grid_scores_]
#        df = pd.concat(rows, axis=1).T.sort([sort_by], ascending=False)
#
#        columns = ['estimator', 'min_score', 'mean_score', 'max_score', 'std_score']
#        columns = columns + [c for c in df.columns if c not in columns]
#
#        return df[columns]

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


def get_reviews(path, infile, corpus, genres=[]):
    '''
    This function gets reviews from a SQLite database, but
    you can substitute your own data retrieval function here.
    Returns a dataframe.
    '''
    print 'reading database'
    conn = sqlite3.connect(os.path.join(path, infile))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("select body, genre, site, book_title, book_author, rating, reviewer_bookshelves "
                "from Reviews "
                "where body <> '' and genre <> '' and site <> ''")  # not empty/null
                # can add LIMIT 5 to do a subset of reviews
    database = [dict(row) for row in cur]
    # database is a list of dicts where {'body': 'text', 'genre': 'genre', 'site': 'site'}
    cur.close()
    conn.close()

    print 'limiting reviews to specific corpus: ' + str(corpus)
    data = []
    for review in database:
        if any (site_name in review['site'] for site_name in corpus):
            if genres:
                for genre in genres:
                    if any (genre_name in review['genre'] for genre_name in genre):
                        review['genre'] = genre[0]
                        soup = BeautifulSoup(review['body'], "lxml")
                        review['body'] = soup.get_text()
                        # remove spacing characters
                        review['body'] = review['body'].replace('\n', ' ')
                        review['body'] = review['body'].replace('\r', ' ')
                        review['body'] = review['body'].replace('\t', ' ')
                        # remove book title from corresponding body (case sensitive) and remove individual capitalized words
                        review['body'] = review['body'].replace(review['book_title'],'')
                        # remove all numbers from body
                        review['body'] = review['body'].replace('\d+', '')
                        # remove genre terms from review body
                        for genre in genres:
                            for genre_name in genre:
                                review['body'] = review['body'].replace(genre_name, ' ')
                        if 'reviewer_bookshelves' in review:
                            if review['reviewer_bookshelves'] == '[]':
                                review['reviewer_bookshelves'] = []
                            else:
                                clean = review['reviewer_bookshelves']
                                clean = clean.split()
                                newshelflist = []
                                for i, item in enumerate(clean):
                                    newitem = item.replace('[u', '').replace("u'", "").replace("'", "")
                                    newitem = newitem.replace(',', '').replace(']', '')
                                    newshelflist.append(newitem)
                                review['reviewer_bookshelves'] = newshelflist
                        data.append(review)
            else:  # if genres is not defined, don't narrow reviews by genre
                soup = BeautifulSoup(review['body'], "lxml")
                review['body'] = soup.get_text()
                # remove spacing characters
                review['body'] = review['body'].replace('\n', ' ')
                review['body'] = review['body'].replace('\r', ' ')
                review['body'] = review['body'].replace('\t', ' ')
                # remove book title from corresponding body (case sensitive) and remove individual capitalized words
                review['body'] = review['body'].replace(review['book_title'],'')
                # remove all numbers from body
                review['body'] = review['body'].replace('\d+', '')
                if 'reviewer_bookshelves' in review:
                    if review['reviewer_bookshelves'] == '[]':
                        review['reviewer_bookshelves'] = []
                    else:
                        clean = review['reviewer_bookshelves']
                        clean = clean.split()
                        newshelflist = []
                        for i, item in enumerate(clean):
                            newitem = item.replace('[u', '').replace("u'", "").replace("'", "")
                            newitem = newitem.replace(',', '').replace(']', '')
                            newshelflist.append(newitem)
                        review['reviewer_bookshelves'] = newshelflist
                data.append(review)

    df = pd.DataFrame(data)
    print df['body']
    return df


def get_reviews_by_year(path, infile, corpus=['goodreadsreviews', 'goodreadsreviewswayback'], genres=[], years=[]):
    '''
    This function gets reviews from a SQLite database by year, but
    you can substitute your own data retrieval function here.
    Returns a dataframe.

    Works for amateur and professional reviews - anything with properly formatted review_pub_date.
    '''
    print 'reading database'
    conn = sqlite3.connect(os.path.join(path, infile))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("select body, genre, site, book_title, book_author, rating, review_pub_date, reviewer_bookshelves "
                "from Reviews "
                "where body <> '' and genre <> '' and site <> ''")  # not empty/null
                # can add LIMIT 5 to do a subset of reviews
    database = [dict(row) for row in cur]
    # database is a list of dicts where {'body': 'text', 'genre': 'genre', 'site': 'site'}
    cur.close()
    conn.close()

    print 'limiting reviews to specific corpus: ' + str(corpus)
    data = []
    for review in database:
        if any (site_name in review['site'] for site_name in corpus):
            if genres:
                for genre in genres:
                    if any (genre_name in review['genre'] for genre_name in genre):
                        review['genre'] = genre[0]
                        soup = BeautifulSoup(review['body'], "lxml")
                        review['body'] = soup.get_text()
                        # remove spacing characters
                        review['body'] = review['body'].replace('\n', ' ')
                        review['body'] = review['body'].replace('\r', ' ')
                        review['body'] = review['body'].replace('\t', ' ')
                        # remove book title from corresponding body (case sensitive) and remove individual capitalized words
                        review['body'] = review['body'].replace(review['book_title'],'')
                        # remove all numbers from body
                        review['body'] = review['body'].replace('\d+', '')
                        # remove genre terms from review body
                        for genre in genres:
                            for genre_name in genre:
                                review['body'] = review['body'].replace(genre_name, ' ')
                        if 'reviewer_bookshelves' in review:
                            if review['reviewer_bookshelves'] == '[]':
                                review['reviewer_bookshelves'] = []
                            else:
                                clean = review['reviewer_bookshelves']
                                clean = clean.split()
                                newshelflist = []
                                for i, item in enumerate(clean):
                                    newitem = item.replace('[u', '').replace("u'", "").replace("'", "")
                                    newitem = newitem.replace(',', '').replace(']', '')
                                    newshelflist.append(newitem)
                                review['reviewer_bookshelves'] = newshelflist
                        data.append(review)
            else:  # if genres is not defined, don't narrow reviews by genre
                soup = BeautifulSoup(review['body'], "lxml")
                review['body'] = soup.get_text()
                # remove spacing characters
                review['body'] = review['body'].replace('\n', ' ')
                review['body'] = review['body'].replace('\r', ' ')
                review['body'] = review['body'].replace('\t', ' ')
                # remove book title from corresponding body (case sensitive) and remove individual capitalized words
                review['body'] = review['body'].replace(review['book_title'],'')
                # remove all numbers from body
                review['body'] = review['body'].replace('\d+', '')
                if 'reviewer_bookshelves' in review:
                    if review['reviewer_bookshelves'] == '[]':
                        review['reviewer_bookshelves'] = []
                    else:
                        clean = review['reviewer_bookshelves']
                        clean = clean.split()
                        newshelflist = []
                        for i, item in enumerate(clean):
                            newitem = item.replace('[u', '').replace("u'", "").replace("'", "")
                            newitem = newitem.replace(',', '').replace(']', '')
                            newshelflist.append(newitem)
                        review['reviewer_bookshelves'] = newshelflist
                data.append(review)

    # trim list of results to only include the relevant year
    data_yr = []
    for review in data:
        try:
            dt = datetime.strptime(review['review_pub_date'], '%Y-%m-%d')
        except:
            try:
                dt = datetime.strptime(review['review_pub_date'], '%m/%d/%Y')
            except:
                # usually this means date was None. set as a date that will never match
                dt = datetime(2025, 1, 1)
        if dt.year in years:
            data_yr.append(review)

    df = pd.DataFrame(data_yr)
    return df


def get_page_n_reviews(path, infile, corpus, genres=[], n=1):
    '''
    This function gets reviews from a SQLite database by the page of
    results the review came from (page 1 through 10), but you can
    substitute your own data retrieval function here.
    Returns a dataframe.
    '''
    print 'reading database'
    conn = sqlite3.connect(os.path.join(path, infile))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("select body, genre, site, book_title, book_author, rating, reviewer_bookshelves, url "
                "from Reviews "
                "where body <> '' and genre <> '' and site <> ''")  # not empty/null
                # can add LIMIT 5 to do a subset of reviews
    database = [dict(row) for row in cur]
    # database is a list of dicts where {'body': 'text', 'genre': 'genre', 'site': 'site'}
    cur.close()
    conn.close()

    print 'limiting reviews to specific corpus: ' + str(corpus)
    data = []
    urlstring = '&from_review_page=' + str(n)
    for review in database:
        if any (site_name in review['site'] for site_name in corpus):
            if review['url'].endswith(urlstring):
                if genres:
                    for genre in genres:
                        if any (genre_name in review['genre'] for genre_name in genre):
                            review['genre'] = genre[0]
                            soup = BeautifulSoup(review['body'], "lxml")
                            review['body'] = soup.get_text()
                            # remove spacing characters
                            review['body'] = review['body'].replace('\n', ' ')
                            review['body'] = review['body'].replace('\r', ' ')
                            review['body'] = review['body'].replace('\t', ' ')
                            # remove book title from corresponding body (case sensitive) and remove individual capitalized words
                            review['body'] = review['body'].replace(review['book_title'],'')
                            # remove all numbers from body
                            review['body'] = review['body'].replace('\d+', '')
                            # remove genre terms from review body
                            for genre in genres:
                                for genre_name in genre:
                                    review['body'] = review['body'].replace(genre_name, ' ')
                            if 'reviewer_bookshelves' in review:
                                if review['reviewer_bookshelves'] == '[]':
                                    review['reviewer_bookshelves'] = []
                                else:
                                    clean = review['reviewer_bookshelves']
                                    clean = clean.split()
                                    newshelflist = []
                                    for i, item in enumerate(clean):
                                        newitem = item.replace('[u', '').replace("u'", "").replace("'", "")
                                        newitem = newitem.replace(',', '').replace(']', '')
                                        newshelflist.append(newitem)
                                    review['reviewer_bookshelves'] = newshelflist
                            data.append(review)
                else:  # if genres is not defined, don't narrow reviews by genre
                    soup = BeautifulSoup(review['body'], "lxml")
                    review['body'] = soup.get_text()
                    # remove spacing characters
                    review['body'] = review['body'].replace('\n', ' ')
                    review['body'] = review['body'].replace('\r', ' ')
                    review['body'] = review['body'].replace('\t', ' ')
                    # remove book title from corresponding body (case sensitive) and remove individual capitalized words
                    review['body'] = review['body'].replace(review['book_title'],'')
                    # remove all numbers from body
                    review['body'] = review['body'].replace('\d+', '')
                    if 'reviewer_bookshelves' in review:
                        if review['reviewer_bookshelves'] == '[]':
                            review['reviewer_bookshelves'] = []
                        else:
                            clean = review['reviewer_bookshelves']
                            clean = clean.split()
                            newshelflist = []
                            for i, item in enumerate(clean):
                                newitem = item.replace('[u', '').replace("u'", "").replace("'", "")
                                newitem = newitem.replace(',', '').replace(']', '')
                                newshelflist.append(newitem)
                            review['reviewer_bookshelves'] = newshelflist
                    data.append(review)

    df = pd.DataFrame(data)
    print df['body']
    return df


def segment_data_by_shelf(df, shelves=[]):
    '''
    limit reviews to reviews on any of a list of given shelves + an equal number not on those shelves
    '''
    # make dataframe of only reviews on any shelf from a list of given shelves
    print 'limiting reviews to those placed on one of the given shelves'
    lim_df = pd.DataFrame(columns=df.columns)
    for index, row in df.iterrows():
        for item in row['reviewer_bookshelves']:
            if item in shelves:
                lim_df.loc[index] = row
            break
    lim_df['shelf_genre'] = 'On shelves'

    # make dataframe of only reviews NOT on any of those shelves, same length as first
    target = lim_df.shape[0]  # get number of reviews in first dataframe to match
    numpy.random.seed(0)
    randf = df.iloc[numpy.random.permutation(len(df))]  # shuffle original data
    # create new dataframe with only reviews NOT on shelves
    other_df = pd.DataFrame(columns=df.columns)
    for index, row in randf.iterrows():
        if other_df.shape[0] < target:  # loop until same length as first dataframe
            has_shelf = False
            for item in row['reviewer_bookshelves']:
                if item in shelves:
                    has_shelf = True
                    break
            if has_shelf == False:
                other_df.loc[index] = row
        else:
            break
    other_df['shelf_genre'] = 'Not on shelves'

    # combine dataframes
    result_df = lim_df.append(other_df)
    return result_df

def print_significant_features(pipeline=None, n=20):
    '''
    This only prints the first class (it's meant for binary classification)
    '''
    feature_names = pipeline.get_params()['vect'].get_feature_names()
    coefs=[]
    try:  # for linear models
        coefs = pipeline.get_params()['clf'].coef_  # attributes in a learner that end with _ are learned
    except:  # for ensemble methods
        coefs.append(pipeline.get_params()['clf'].feature_importances_)
    print "Total features: %d" % (len(coefs[0]))
    coefs_with_fns = sorted(zip(coefs[0], feature_names))
    top = coefs_with_fns[:-(n + 1):-1]
    for (coef_2, fn_2) in top:
        print "%f: %s" % (coef_2, str(fn_2))


def print_significant_features_by_class(pipeline=None, X=None, Y=None, n=20):
    '''
    Linear models: Prints top (and bottom if meaningful) feature importance coefficients

    Ensemble models:
    Takes a matrix of documents x tfidf features
    Multiplies by the feature importance coefficients of the model overall, by class
    Prints the top n products, showing words that appeared most for that class
    and were of high importance to the model.
    '''
    feature_names = pipeline.get_params()['vect'].get_feature_names()
    # get feature importance coefficients
    # for linear models with meaningful pos and neg coefficients:
    linear_models = (LogisticRegressionCV, LinearSVC)
    if isinstance(pipeline.get_params()['clf'], linear_models):
        coefs = pipeline.get_params()['clf'].coef_  # attributes in a learner that end with _ are learned
        if len(coefs) == 1:  # binary classifier
            cls_names = pipeline.get_params()['clf'].classes_
            coefs_with_fns = sorted(zip(pipeline.get_params()['clf'].coef_[0], feature_names))
            print 'Predicts ' + cls_names[1]
            for item in coefs_with_fns[:-(n + 1):-1]:  # highest n scores
                print str(item[0]), item[1]
            print ''
            print 'Predicts ' + cls_names[0]
            for item in coefs_with_fns[:n]:  # lowest n scores
                print str(item[0]), item[1]
        else:  # multiclass classifier
            # format of coefs: matrix with rows = to number of classes,
            # columns = to number of features, each entry is coef of feature importance
            cls_names = pipeline.get_params()['clf'].classes_
            full = {}
            for i, cls in enumerate(coefs):
                full[cls_names[i]] = sorted(zip(cls, feature_names))
            top = {}
            for key, value in full.iteritems():
                top[key] = value[:-(n + 1):-1]
            bottom = {}
            for key, value in full.iteritems():
                bottom[key] = value[:n]
            for k, v in top.iteritems():
                print 'Predicts ' + str(k)
                for item in v:
                    print unicode(item[0]) + ' ' + unicode(item[1])
                print ''
            print ''
            for k, v in bottom.iteritems():
                print 'Predicts NOT ' + str(k)
                for item in v:
                    print unicode(item[0]) + ' ' + unicode(item[1])
                print ''
            print ''

    # for linear models that only have positive coefficients
    coef_models = (MultinomialNB)
    if isinstance(pipeline.get_params()['clf'], coef_models):
        coefs = pipeline.get_params()['clf'].coef_  # attributes in a learner that end with _ are learned
        # format of coefs: matrix with rows = to number of classes,
        # columns = to number of features, each entry is coef of feature importance
        cls_names = pipeline.get_params()['clf'].classes_
        full = {}
        for i, cls in enumerate(coefs):
            full[cls_names[i]] = sorted(zip(cls, feature_names))
        top = {}
        for key, value in full.iteritems():
            top[key] = value[:-(n + 1):-1]
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(top)

    # for ensemble models:
    fi_models = (ExtraTreesClassifier, DecisionTreeClassifier, 
                 RandomForestClassifier, GradientBoostingClassifier)
    if isinstance(pipeline.get_params()['clf'], fi_models):
        coefs = pipeline.get_params()['clf'].feature_importances_
        X2 = pipeline.named_steps['vect'].fit_transform(X)
        X3 = X2.todense()
        X3 = numpy.squeeze(numpy.asarray(X3))
        full = {}
        for c in set(Y):
            full[c] = dict(zip(feature_names, numpy.mean(X3[Y==c, :], axis=0)*coefs))
        out = {}
        for key in full:
            out[key] = sorted(full[key].iteritems(), key=operator.itemgetter(1), reverse=True)[:n]
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(out)


def plot_significant_features(pipeline=None, n=20):
    feature_names = pipeline.get_params()['vect'].get_feature_names()
    coefs=[]
    try:
        coefs = pipeline.get_params()['clf'].coef_
    except:
        coefs.append(pipeline.get_params()['clf'].feature_importances_)
    
    print "Total features: %d" % (len(coefs[0]))
    coefs_with_fns = sorted(zip(coefs[0], feature_names))
    top = coefs_with_fns[:-(n + 1):-1]
    
    y,X = zip(*top)

    plt.figure()
    plt.title("Top 20 most important features")
    plt.gcf().subplots_adjust(bottom=0.25)
    ax = plt.subplot(111)
    
    ax.bar(range(len(X)), y, color="r", align="center")
    ax.set_xticks(range(len(X)))
    ax.set_xlim(-1, len(X))
    ax.set_xticklabels(X,rotation='vertical')
    plt.savefig('sentiment_feature_importance.png')
    plt.close()


def grid_search(pipeline, X, y):
    '''
    Takes an extremely long time
    '''
    parameters = {
    #    'vect__strip_accents': ('ascii', 'unicode', None),
        'vect__tokenizer': (None, LemmaTokenizer()),
        'vect__ngram_range': ((1,1), (1,2)),
    #    'vect__stop_words': stops,
        'vect__lowercase': (True, False),  # Convert all characters to lowercase before tokenizing.
        'vect__max_df': (0.6, 0.9),  # automatic stopwords
        'vect__min_df': (1, 3),
    #    'vect__max_features': 10000,
        'vect__binary': (True, False),  # If True, all non-zero term counts are set to 1.
    #    'vect__norm': (None, 'l1', 'l2')
        'vect__use_idf': (True, False),
        'feature_selection__k': (1000, 'all'),
    }

    grid_search = GridSearchCV(pipeline, parameters, n_jobs=1, verbose=100, cv=10)
    grid_search.fit(X, y)

    print "Best score: %0.3f" % grid_search.best_score_
    print "Best parameters set:"
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print "\t%s: %r" % (param_name, best_parameters[param_name])

    # for grid searching classifiers
    #models = {
    #    'DecisionTreeClassifier': DecisionTreeClassifier(),
    #    'RandomForestClassifier': RandomForestClassifier(),
    #    'ExtraTreesClassifier': ExtraTreesClassifier(),
    #    'MultinomialNB': MultinomialNB(),
    #    'LogisticRegressionCV': LogisticRegressionCV(),
    #    'LinearSVC': LinearSVC(),
    #}
    #
    #params = {
    #    'DecisionTreeClassifier': {},
    #    'RandomForestClassifier': {},
    #    'ExtraTreesClassifier': {},
    #    'MultinomialNB': {},
    #    'LogisticRegressionCV': {},
    #    'LinearSVC': {},
    #}


def make_predictions(pipeline, X, y):
    '''
    Predicts the class of reviews.
    '''
    print 'making predictions'
    t0 = time()

    pipeline.fit(X, y)
    print "Training done in %0.3fs" % (time() - t0)
    print ''

    predictions = cross_val_predict(pipeline, X, y)
    predictions.shape
    print classification_report(y, predictions)
    print 'precision = true positives / (true positives + false positives)'
    print 'recall = true positives / (true positives + false negatives)'
    print 'f1-score = weighted mean of precision and recall (1 is best, 0 is worst)'
    print 'support = number of occurrences of each class in y_true'
    print ''

    # for anything that has coef_ or feature_importances_
    safetypes = (ExtraTreesClassifier, DecisionTreeClassifier, MultinomialNB, 
                 LogisticRegressionCV, RandomForestClassifier, LinearSVC)
    if isinstance(pipeline.get_params()['clf'], safetypes):
        print_significant_features_by_class(pipeline=pipeline, X=X, Y=predictions)

    print "Total done in %0.3fs" % (time() - t0)


if __name__ == '__main__':
    path = "E:/0-docs/diss/data/"
    infile = "diss.sqlite"

    # which sites correspond to which review category
    professional = ["austinchronicle", "bostonreview", "kirkus", "larb", 
                    "proquest", "proquestwsj", "proquestlat", "proquestnyt",
                    "publishersweekly", "usatodayreviews", "wirb"]
    amateur = ["goodreadsreviews", "goodreadsreviewswayback", "goodreadsreviewsonshelves"]
    academic = ["projectmuse"]

    # possible genre names from review json files for each of the genres I'm interested in
    sf = ["Science Fiction", "Science fiction", "science fiction", "science-fiction", "SF", "sf", "Fantasy/Sci-fi", "Science Fiction & Fantasy", "Science fiction & fantasy", "scifi", "sci fi", "sci-fi"]
    horror = ["Horror", "horror"]
    romance = ["Romance", "romance", "romances"]
    detective = ["Mystery", "Mysteries", "Detective Fiction", "detective fiction", "noir", "Thrillers", "thrillers", "thriller", "Mystery & Crime", "Mystery & Suspense", "Suspense", "suspense", "mystery", "mysteries", "Crime Fiction", "crime fiction", "crime", "Crime"]
    biography = ["Biography", "biography & autobiography", "Memoir", "memoir", "memoirs", "Biographies", "Biography & Memoir", "biography", "biographies", "biographical", "biographer", "biographers", "autobiographical", "hagiography", "Autobiographies", "autobiographies", "Autobiography", "autobiography"]
    history = ["History", "history", "histories"]
    genres = [sf, horror, romance, detective, biography, history]

    print 'getting data'
    # uncomment the method you want to use
#    df = get_reviews(path, infile, amateur, genres=genres)  # uses reviews from trad genre corpus
#    df = get_reviews(path, infile, amateur, genres=[])  # uses all reviews (for narrowing by review shelf)
#    df = get_reviews_by_year(path, infile, professional, genres=genres, years=[1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015])  # for trad genres
    df = get_reviews_by_year(path, infile, amateur, genres=[], years=[2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016])  # for shelf groups
#    df = get_reviews_by_year(path, infile, academic, genres=genres, years=[1973, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016])

    # compare F1 score of Goodreads results page 1 reviews to page 2 reviews (n=page)
#    df = get_page_n_reviews(path, infile, amateur, genres=genres, n=1)

    shelves = ['wwii', 'world-war-ii', 'world-war-2', 'ww-ii', 'world-war-two',
               'ww2', 'worldwarii']
    shelves = ['creepy']
    print shelves
    df = segment_data_by_shelf(df, shelves=shelves)
    print df

    # FOR WORKING WITH GENRE CORPUS (NOT ALL REVIEWS)
    # make it binary between one class and the rest
#    df['genre'] = df['genre'].replace(['Biography', 'Romance', 'History',
#                                      'Science Fiction', 'Mystery'], 'Not Horror')
#    # take random sample from larger class to equalize group sizes
#    size = df['genre'].value_counts()['Horror']  # get size of target class
#    numpy.random.seed(0)
#    fn = lambda obj: obj.loc[numpy.random.choice(obj.index, size, False), :]
#    df = df.groupby('genre', as_index=False).apply(fn)
    # print number of unique books reviewed (only for amateur/goodreads and gives partial total for prof)
    print 'unique books: ' + str(len(df.book_title.unique()))


    X = df['body'].values
    if 'shelf_genre' in df:  # check if you're classifying a shelf or a traditional genre
        y = df['shelf_genre']
    else:
        y = df['genre']

    # make list of author names to use as stopwords
    authors = set(df['book_author'])
    names = set()
    for author in authors:
        l = author.split()
        names.update(l)

    stops = set(stopwords.words("english"))
    stops.update(names)
    try:
        stops.update(shelves)
    except:
        pass
    stops.update(['http', 'br', 'tr'])
    stops.update(['hitler', 'booth', 'lewis', 'zamperini', 'louie', 'churchill', 'lestat', 'zippy', 'bateman', 'bates'])
    stops.update(['morrie', 'marley', 'mccandless', 'corfu', 'shackleton', 'einstein', 'martha', 'roosevelt', 'lbj'])
    stops.update(['roland', 'ripley', 'reacher', 'bond', 'marlowe', 'courtney', 'cordelia', 'chigurh', 'kinsey', 'himmler'])
    stops.update(['robicheaux', 'marian', 'sherlock', 'bourne', 'claire', 'scarlett', 'heathcliff', 'bridget', 'kennedys'])
    stops.update(['bronte', 'darcy', 'tris', 'jacob', 'cassia', 'rochester', 'bella', 'ruby', 'romeo', 'auden', 'ender'])
    stops.update(['katniss', 'foundation', 'kivrin', 'emma', 'winston', 'macarthur', 'camille', 'nick', 'lecter', 'eve'])
    stops.update(['powell', 'raoul', 'potter', 'amy', 'christine', 'manderley', 'harry', 'dresden', 'lena', 'mapplethorpe'])
    stops.update(['ian', 'sybil', 'teddy', 'theodore', 'cleopatra', 'rosemary', 'millhone', 'holmes', 'mcevoy', 'brigance'])
    stops.update(['roarke', 'rhett', 'cora', 'hara', 'evan', 'bennet', 'taryn', 'maddox', 'offred', 'dunworthy', 'eleanor'])
    stops.update(['ash', 'sam', 'ibal', 'clarice', 'patrick', 'peeta', 'ky', 'scarpetta', 'oprah', 'caesar', 'jake', 'bosch'])
    stops.update(['kay', 'nate', 'jane', 'hazel', 'taylor', 'starling', 'nora', 'xander', 'crawford', 'sean', 'maus', 'aubrey'])
    stops.update(['shamus', 'iii', 'salander', 'tess', 'himes', 'borges', 'fisher', 'sayers', 'wright', 'chesterton'])
    stops.update(['barber', 'vian', 'chandler', 'arnold', 'dupin', 'malory', 'ophelia', 'pocahontas', 'hutchinson', 'tomas'])
    stops.update(['edna', 'meyer', 'havelok', 'trilby', 'vera', 'morris', 'perceval', 'briony', 'saville', 'othello'])
    stops.update(['ralph', 'clarel', 'curial', 'vankirk', 'camino', 'buffy', 'gaiman', 'medea', 'ginger', 'thacker'])
    stops.update(['bruchac', 'wendigo', 'sacher', 'masoch', 'bechdel', 'joe', 'bachmann', 'kubrick', 'dain', 'boyd'])
    stops.update(['defoe', 'napoleon', 'dorian', 'melville', 'pierre', 'harley', 'des', 'peter', 'pan', 'owens', 'wendy'])
    stops.update(['wells', 'delany', 'butler', 'shiel', 'frankenstein', 'utopia', 'oedipus', 'cavell', 'pap', 'mccarthy'])
    stops.update(['silberrad', 'laurens', 'bertillon', 'oscar', 'parker', 'collins', 'tolkien', 'davis', 'jeanne', 'jung'])
    stops.update(['smith', 'justine', 'clarel', 'malet', 'le', 'helen', 'winterson', 'pp', 'rose', 'spenser', 'christopher'])
    stops.update(['el', 'spark', 'archer', 'verloc', 'atalanta', 'judy', 'nabokov', 'browning', 'svengali', 'mehta', 'chen'])
    stops.update(['galahad', 'rossetti', 'glenthorn', 'memed', 'fleda', 'almayer', 'lingard', 'charlie', 'chopin', 'bod'])
    stops.update(['brigitte', 'snowman', 'trier', 'jason', 'mowgli', 'tolstoy', 'barr', 'dana', 'williams', 'bloom'])
    stops.update(['lombroso', 'padura', 'montaigne', 'george', 'dickens', 'yunior', 'behn', 'poe', 'ti', 'sophocles', 'de'])
    stops.update(['beckert', 'vivian', 'sebald', 'brautigan', 'bolano', 'vrkljan', 'sebastian', 'kemal', 'launfal', 'gawan'])
    stops.update(['mahony', 'cobham', 'willems', 'eaton', 'loftus', 'chretien', 'von', 'kipling', 'beatrice', 'austen'])
    stops.update(['dreiser', 'henry', 'rudy', 'la', 'lang', 'hamlet', 'dora', 'acs', 'prout', 'rupert', 'clover', 'cabrera'])
    stops.update(['diaz', 'delillo', 'auster', 'chabon', 'meade', 'greg', 'moretti', 'sterne', 'lovecraft', 'fol', 'baojuan'])
    stops.update(['arabella', 'fielding', 'gauvain', 'wao', 'eisner', 'addison', 'lu', 'woolf', 'johnson', 'jovanovic'])
    stops.update(['ryan', 'bob', 'scott', 'bowen', 'rorty', 'frank', 'flora', 'king', 'bill', 'shojo', 'beckett', 'cohen'])
    stops.update(['benedict', 'jovanovic', 'risco', 'ken', 'raffi', 'las', 'wieder', 'greene', 'machen', 'op', 'rand'])
    stops.update(['sedgwick', 'hawthorne', 'pound', 'twain', 'frye', 'dai', 'orlando', 'digby', 'brenner', 'thoreau'])
    stops.update(['katharine', 'fu', 'marguerite', 'harrington', 'violet', 'douglas', 'manchu', 'anna', 'augustus', 'ful'])
    stops.update(['rhage', 'pkd', 'engle', 'jonas', 'kelsey', 'jones', 'johnny', 'palmer', 'poirot', 'dickie', 'lisbeth'])
    stops.update(['meg', 'roberts', 'landon', 'juliet', 'mayfair', 'blomkvist', 'graham', 'smersh', 'wool', '1984'])
    stops.update(['pendergast', 'lincoln', 'flagg', 'sk', 'alex', 'twilight', 'triffids', 'hannibal', 'moreau', 'prendick'])
    stops.update(['shining', 'torrance', 'danny', 'stand', 'gabriel', 'dallas', 'geisha', 'sayuri', 'chiyo', 'pnr', 'archie'])
    stops.update(['moss', 'kovacs', 'bancroft', 'persepolis', 'scissors', 'lukacs', 'bersani', 'erauso', 'campos', 'baldwin'])
    stops.update(['freud', 'kluger', 'zorn', 'anne', 'santos', 'coraline', 'shklovsky', 'hemingway', 'welty', 'nancy'])
    stops.update(['hailsham', 'jasmine', 'robinson', 'susan', 'middlemarch', 'kouty', 'stedman', 'mieville', 'drew'])
    stops.update(['faulkner', 'que', 'fig', 'quixote', 'calinescu', 'lou', 'eliot', 'dowell', 'alcott', 'julian', 'segalla'])
    stops.update(['derrida', 'colpeper', 'faolain', 'macardle', 'flaubert', 'alkan', 'malkiel', 'juno', 'broughton'])
    stops.update(['seinfeld', 'conrad', 'khansa', 'lery', 'steinbeck', 'lucrece', 'dojinshi', 'joyce', 'helva', 'demy'])
    stops.update(['caithleen', 'pullman', "khansaʾ", 'crusoe', 'eudora', 'shlomo', 'cappiello', 'aue', 'gawain', 'merrick'])
    stops.update(['burroughs', 'stanwyck', 'guin', 'alice', 'stein', 'adams', 'fitzgerald', 'stevens', 'littell', '2013'])
    stops.update(['cavendish', 'vendler', 'gorgik', 'gatsby', 'seimei', 'michelson', 'josie', 'miranda', 'succes', 'ul'])
    stops.update(['bobby', 'chablis', 'hermione', 'anita', 'azimov', 'zoey', 'mercutio', 'lee', 'carl', 'mortensen', 'liz'])
    stops.update(['henrietta', 'kate', 'flds', 'catherine', 'heidegger', 'kathy', 'hal', 'bean', 'luce', 'leguin', 'silo'])
    stops.update(['sci', 'fi', 'watney', 'hyde', 'mike', 'jamie', 'ig', 'doug', 'neill', 'jenna', 'jude', 'merricat'])
    stops.update(['ashlyn', 'dexter', 'clary', 'remy', 'wes', 'beth', 'macy', 'fwai', 'payton', 'louis', 'cam', 'kiera'])
    stops.update(['travis', 'royce', 'ka', 'jj', 'wolfe', 'jace', 'marvin', 'melanie', 'davinci', 'gamache', 'baigent'])
    stops.update(['da', 'carre', 'lynley', 'faber', 'spade', 'arkady', 'mikael', 'nero', 'renko', 'cameron', 'reich'])
    stops.update(['tom', 'leamas', 'cathy', 'dalgliesh', 'cain', 'vinci', 'dave', 'mish', 'goldfinch', 'maggie', 'walter'])
    stops.update(['greenleaf', 'annette', 'marie', 'antoinette', 'clark', 'columbus', 'eric', 'tayo', 'abigail', 'jekyll'])
    stops.update(['caligula', 'ayn', 'sf', 'scifi', 'haley', 'max', 'phoebe', 'nina', 'peyton', 'toland', 'ashton', 'toph'])
    stops.update(['jean', 'caleb', 'stewart', 'beau', 'layken', 'joss', 'sawyer', 'arnie', 'lenny', 'enders', 'pournelle'])
    stops.update(['herbert', 'lucy', 'shevek', 'mick', 'jd', 'moroi', 'neal', 'edward', 'henrik', 'fdr', 'relin', 'ella'])
    stops.update(['goll', 'reagan', 'finley', 'lenin', 'ellroy', 'redstone', 'redmond', 'dodgson', 'evans', 'meinertzhagen'])
    stops.update(['fleming', 'cheever', 'montague', 'thurston', 'trotsky', 'podhoretz', 'rembrandt', 'novak', 'lerman'])
    stops.update(['welles', 'richelieu', 'soto', 'winn', 'bruni', 'rosenfeld', 'carey', 'crosby', 'mauldin', 'kreuger'])
    stops.update(['reven', 'cosima', 'kelly', 'ford', 'schama', 'swanson', 'mellon', 'montefiore', 'chavez', 'webster'])
    stops.update(['clinton', 'itzkoff', 'dickinson', 'philip', 'gretchen', 'stahl', 'heinrich', 'ali', 'einhorn', 'dickerson'])
    stops.update(['mcluhan', 'wasserman', 'steffens', 'daley', 'rescorla', 'marshall', 'jackson', 'cramer', 'chanel', 'jett'])
    stops.update(['burr', 'willis', 'vanderbilt', 'sharon', 'dimaggio', 'yogi', 'ransome', 'bush', 'maynard', 'baker'])
    stops.update(['hamilton', 'newton', 'highsmith', 'witte', 'schenkar', 'boswell', 'ziegfeld', 'mitchell', 'foote', 'stalin'])
    stops.update(['lapidus', 'tati', 'rawlins', 'cassavetes', 'nixon', 'coe', 'eileen', 'auchincloss', 'spalding', 'silverman'])
    stops.update(['wolper', 'peterson', 'seaver', 'ravitch', 'florman', 'claiborne', 'thackeray', 'bogart', 'matisse', 'maugham'])
    stops.update(['mozart', 'mcnall', 'eichmann', 'copernicus', 'leonowens', 'franklin', 'mcclellan', 'hayek', 'heydrich'])
    stops.update(['murdoch', 'galbraith', 'kissinger', 'parsons', 'ann', 'rushdie', 'wodehouse', 'sade', 'richenbacker'])
    stops.update(['mastrogiacomo', 'amis', 'andrews', 'malamud', 'clemente', 'murray', 'gershwin', 'rickenbacker', 'spearman'])
    stops.update(['harvey', 'preminger', 'mao', 'macaulay', 'mishima', 'yates', 'dickstein', 'philby', 'sinatra', 'bailey'])
    stops.update(['weill', 'hampton', 'massie', 'mitford', 'alsop', 'gandhi', 'palewski', 'madison', 'kitchener', 'molineux'])
    stops.update(['frick', 'taney', 'babchenko', 'wilson', 'kennan', 'kosciuszko', 'santayana', 'greeley', 'douglass', 'yeltsin'])
    stops.update(['olmsted', 'thompson', 'dorothy', 'muriel', 'alexander', 'liu', 'koch', 'disch', 'patterson', 'lasker'])
    stops.update(['palmerston', 'hartigan', 'ross', 'dixon', 'lilley', 'mokyr', 'lyons', 'mercedes', 'feynman', 'altman'])
    stops.update(['daniels', 'koryta', 'fred', 'cooper', 'mordden', 'constance', 'weintraub', 'kagan', 'sati', 'vermeer'])
    stops.update(['giuliani', 'spitzer', 'carlin', 'greenspan', 'rebus', 'derek', 'dick', 'holt', 'phyllis', 'val', 'jimmy'])
    stops.update(['toni', 'blaire', 'juliette', 'tamani', 'sookie', 'sydney', 'cliffy', 'amelia', 'warner', 'abby', 'olivia'])
    stops.update(['charley', 'esther', 'alisa', 'ukiah', 'knulp', 'chaya', 'mara', '12', 'dex'])
    stops = set(x.lower() for x in stops)

    pipeline = Pipeline([
#        ('vect', TfidfVectorizer(strip_accents='unicode', stop_words=stops, max_features=10000, norm='l2')),  # for grid search
        ('vect', TfidfVectorizer()),  # for classifying
#       ('clf', DecisionTreeClassifier()),
#       ('clf', RandomForestClassifier()),
#       ('clf', ExtraTreesClassifier(n_jobs=-1, n_estimators=500)),
#       ('clf', MultinomialNB()),
#       ('clf', LogisticRegressionCV()),
        ('clf', LinearSVC()),
    ])

#    grid_search(pipeline, X, y)  # 2560 jobs: 7216.5min

    # Use parameters with best results from grid search
    pipe_params = {
    #    'clf__class_weight': 'balanced',
    #    'clf__random_state': 42,
        'vect__strip_accents': 'unicode',
        'vect__norm': 'l2',
        'vect__use_idf': True,
        'vect__max_df': 0.6,
        'vect__min_df': 1,
        'vect__max_features': 10000,
        'vect__ngram_range': (1,1),
        'vect__stop_words': stops,
        'vect__lowercase': True,
        'vect__binary': False,
    }

    pipeline.set_params(**pipe_params)

    make_predictions(pipeline, X, y)
