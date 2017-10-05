# -*- coding: utf-8 -*-
"""
This program takes a json file of goodreads shelf data (created by
goodreadsshelves or goodreadsshelveswayback spider) and analyzes it
as a Pandas DataFrame.

It clusters shelves into groups based on the books people have placed
them on. Output is printed scatterplots and (for hierarchical clustering)
a PNG of the cluster dendrogram.
"""

import json
import operator
from pandas import DataFrame
import pandas as pd
import string
from datetime import datetime
import sqlite3
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.externals import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import manifold, decomposition
from collections import Counter
from scipy.cluster.hierarchy import linkage
from heapq import nsmallest
from sklearn.metrics import silhouette_samples, silhouette_score
import hdbscan
import seaborn as sns


def get_shelf_data(shelf_json, max_ratio=0.25, min_books_using_shelf=1):
    '''
    shelf_json is a json file for a book from Goodreads.com with information
    about the shelves users have placed it on.
    Its format should be json where {'book_title': title, 'shelf': #}
    max_ratio is the maximum % of books any given shelf can use (gets rid of super popular shelves like 'to-read')
    min_books_using_shelf is the minimum books a shelf must have to be included
    '''
    print 'getting shelf data'
    data = []

    with open(shelf_json) as f:
        for i, line in enumerate(f):
            if i < 8000:
                if not line.startswith(('[', ']')):
                    if line.endswith(',\n'):
                        data.append(json.loads(line[:-2]))
                    else:
                        data.append(json.loads(line))

#    print get_top_shelves(data)

    valid_chars = set(string.printable)
    # add to this set to include characters from other languages

    raw_shelfcounts = []
    shelfcounts = []
    for book in data:
        # only include book if title is all English letters
        if all(letter in valid_chars for letter in book['book_title'][0]):
            bookdict = {}
            # get total number of shelves the book has been placed on
            total_shelves = float(book['total_shelves'][0].split(' of ')[1].replace('\n', '').replace(',', '')[:-1])

            # get list of all the shelf names for the current book
            shelflist = book['shelf']
            total_count = 0.0
            for i, shelf in enumerate(shelflist):  # connects the people counts (as integers) to the correct shelf names as key:value pairs
                if all(letter in valid_chars for letter in shelf):  # only include shelf if name is all English letters
                    peoplecount = book['people'][i].replace('\n', '').replace('\t', '').strip()
                    # remove commas and " people" at the end of the count
                    peoplecount = float(peoplecount[:-7].replace(',', ''))
                    if peoplecount > 0:  # goodreads data is messy, sometimes has -1 people
                        total_count += peoplecount
                        bookdict[shelf] = peoplecount

            # normalize shelf counts by each book's total shelvings
            norm_bookdict = {}
            for shelf, count in bookdict.iteritems():
                if count < total_count:  # goodreads data is messy, sometimes total shelf count is too small
                    norm_bookdict[shelf] = count/total_count  # turn shelf counts into % of total shelvings
                elif count < total_shelves:
                    norm_bookdict[shelf] = count/total_shelves

            norm_bookdict['book_title'] = book['book_title'][0].replace('\n', '').strip()
            shelfcounts.append(norm_bookdict)
            bookdict['book_title'] = book['book_title'][0].replace('\n', '').strip()
            raw_shelfcounts.append(bookdict)

    # trim list of shelves to remove shelves that house many books (less meaningful)
    # this is like the IDF part of TF-IDF
    # create dataframe of all the counts for easier calculating
    # with rows as shelves and columns as book titles
    ratio_df = DataFrame(shelfcounts).fillna(0).set_index('book_title').T
    # for each shelf, calculate the proportion of total books that were placed on it
    # make dataframe of the number of non-zero entries for each shelf
    # total columns minus number of columns with value zero
    nonzero_df = (len(ratio_df.columns) - (ratio_df == 0).astype(float).sum(axis=1))/len(ratio_df.columns)
    ratio_dict = nonzero_df.to_dict()
    sel_shelfcounts = []
    for book in shelfcounts:
        bookdict = {}
        bookdict['book_title'] = book['book_title']
        del book['book_title']
        # only include shelf if below a certain ratio of books use that shelf
        for shelf, score in book.iteritems():
            ratio = ratio_dict[shelf]
            # ignore shelves with too many and too few uses
            if ratio < max_ratio and ratio > (float(min_books_using_shelf)/len(ratio_df.columns)):
                bookdict[shelf] = score
            else:
#                print 'denied ' + shelf + ' with ratio ' + str(ratio)
                continue
        sel_shelfcounts.append(bookdict)

    # trim list of shelves to only the top n shelves for each book
    sel2_shelfcounts = []
    for book in sel_shelfcounts:
        bookdict = {}
        bookdict['book_title'] = book['book_title']
        del book['book_title']
        # gets highest n scores - 10 because that's how many show up on main book page on Goodreads
        for shelf, score in nsmallest(10, book.iteritems(), key=lambda (k, v): (-v, k)):
            bookdict[shelf] = score
        sel2_shelfcounts.append(bookdict)

    # do the same for the raw counts
    raw_sel2_shelfcounts = []
    for book in raw_shelfcounts:
        bookdict = {}
        bookdict['book_title'] = book['book_title']
        del book['book_title']
        # gets highest n scores - 10 because that's how many show up on main book page on Goodreads
        for shelf, score in nsmallest(10, book.iteritems(), key=lambda (k, v): (-v, k)):
            bookdict[shelf] = score
        raw_sel2_shelfcounts.append(bookdict)

    return DataFrame(sel2_shelfcounts).fillna(0).set_index('book_title'), \
           DataFrame(raw_sel2_shelfcounts).fillna(0).set_index('book_title')


def get_wayback_shelf_data(shelf_json, max_ratio=0.25, min_books_using_shelf=1, n=10, years=[]):
    '''
    shelf_json is a json file for a previous snapshot of a book from
    Goodreads.com obtained from the Wayback Machine, with information
    about the shelves users have placed it on.
    Its format should be json where {'book_title': title, 'shelf': #}
    max_ratio is the maximum % of books any given shelf can use (gets rid of super popular shelves like 'to-read')
    min_books_using_shelf is the minimum books a shelf can have to still be included
    n: returns highest n shelves for each book (10 is what shows up on Goodreads book page)
    years is a list of integer years to include data from
    '''
    print 'getting wayback shelf data'
    data = []

    with open(shelf_json) as f:
        for line in f:
            if not line.startswith(('[', ']')):
                if line.endswith(',\n'):
                    data.append(json.loads(line[:-2]))
                else:
                    data.append(json.loads(line))

#    print get_top_shelves(data)

    valid_chars = set(string.printable)
    # add to this set to include characters from other languages

    raw_shelfcounts = []
    shelfcounts = []
    for book in data:
        for i, title in enumerate(book['book_title']):
            if title.isspace():
                del book['book_title'][i]
        if 'book_title' not in book:
            continue
        if len(book['book_title']) > 1:
            print 'book_title rejected because more than 1'
            print book['book_title']
            continue
        # this deals with mistakes in scraping where people is empty but
        # shelf has e.g. ['to-read (586)', 'currently-reading (148)']
        if 'shelf' in book and 'people' in book:
            if book['shelf'] and not book['people']:
                # only change shelves with both '(' and ')' in every shelf name
                if all('(' in x for x in book['shelf']) and all(')' in x for x in book['shelf']):
                    for k, item in enumerate(book['shelf']):
                        both = item.split()
                        book['shelf'][k] = both[0]
                        book['people'].append(both[1].replace('(', '').replace(')', ''))

        if book['total_shelves'] == ['by']:  # not sure how this happened in scraper; cleanup
            book['total_shelves'] = 1

        if book['people'] and book['shelf'] and book['url'] and len(book['people']) == len(book['shelf']):
            book['book_title'][0] = book['book_title'][0].replace('\n', '').strip()
            # only include book if title is all English letters
            if all(letter in valid_chars for letter in book['book_title'][0]):
                bookdict = {}
                # get total number of shelves the book has been placed on
                # source 1: from goodreads count; this is sometimes smaller than individual shelf counts
                if type(book['total_shelves']) == list:
                    total_shelves = float(book['total_shelves'][0].split(' of ')[1].replace('\n', '').replace(',', '')[:-1])
                elif type(book['total_shelves']) == int:
                    total_shelves = float(book['total_shelves'])

                # get list of all the shelf names and counts for the current book
                shelflist = book['shelf']
                total_count = 0.0
                for i, shelf in enumerate(shelflist):  # connects the people counts (as integers) to the correct shelf names as key:value pairs
                    shelf = shelf.lower().replace(' ', '-')  # make sure all shelves follow same pattern of all-lowercase with dashes instead of spaces (Goodreads changed format sometime 2015ish)
                    if all(letter in valid_chars for letter in shelf):  # only include shelf if name is all English letters
                        peoplecount = book['people'][i].replace('\n', '').replace('\t', '').strip()
                        # remove commas and " people"/" users" at the end of the count
                        if 'ves)' in peoplecount:  # '(on 93 people's shelves)'
                            peoplecount = peoplecount.replace('(on ', '').replace(" people's shelves)", "").replace(',', '')
                        elif 'f)' in peoplecount:  # '(on 1 person's shelf)'
                            peoplecount = peoplecount.replace('(on ', '').replace(" person's shelf)", "").replace(',', '')
                        elif 'pe' in peoplecount:
                            peoplecount = peoplecount[:-7].replace(',', '')
                        elif 'users' in peoplecount or 'shelf' in peoplecount:
                            peoplecount = peoplecount[:-6].replace(',', '')
                        elif 'user' in peoplecount:
                            peoplecount = peoplecount[:-5].replace(',', '')
                        elif 'shelves' in peoplecount:
                            peoplecount = peoplecount[:-8].replace(',', '')
                        peoplecount = float(peoplecount)
                        if peoplecount > 0:  # goodreads data is messy, sometimes has -1 people
                            total_count += peoplecount
                            bookdict[shelf] = peoplecount

                # normalize shelf counts by each book's total shelvings
                norm_bookdict = {}
                for shelf, count in bookdict.iteritems():
                    if count < total_count:  # goodreads data is messy, sometimes total shelf count is too small
                        norm_bookdict[shelf] = count/total_count  # turn shelf counts into % of total shelvings
                    elif count < total_shelves:
                        norm_bookdict[shelf] = count/total_shelves

                # bookdict is un-normalized raw counts; norm_bookdict is normalized
                bookdict['book_title'] = book['book_title'][0].replace('\n', '').strip()
                norm_bookdict['book_title'] = book['book_title'][0].replace('\n', '').strip()

                # add date and time of shelfcount snapshot from Wayback Machine URL
                timestamp = book['url'].split('/')[4]
                dt = datetime(int(timestamp[:4]), int(timestamp[4:6]), int(timestamp[6:8]),
                              int(timestamp[8:10]), int(timestamp[10:12]), int(timestamp[12:14]))
                bookdict['datetime'] = dt
                norm_bookdict['datetime'] = dt

                raw_shelfcounts.append(bookdict)
                shelfcounts.append(norm_bookdict)

    # trim list of results to only include the relevant year(s)
    shelfcounts_yr = []
    raw_shelfcounts_yr = []
    for book in shelfcounts:
        if book['datetime'].year in years:
            # don't add duplicates of the same book - only one per year
            if not any(d['book_title'] == book['book_title'] for d in shelfcounts_yr):
                del book['datetime']
                shelfcounts_yr.append(book)
    if not shelfcounts_yr:
        print 'NO REVIEWS FOR THAT YEAR; TRY A DIFFERENT YEAR'
    for book in raw_shelfcounts:
        if book['datetime'].year in years:
            # don't add duplicates of the same book - only one per year
            if not any(d['book_title'] == book['book_title'] for d in raw_shelfcounts_yr):
                del book['datetime']
                raw_shelfcounts_yr.append(book)

    # trim list of shelves to remove shelves that house many books (less meaningful)
    # this is like the IDF part of TF-IDF
    # create dataframe of all the counts for easier calculating
    # with rows as shelves and columns as book titles
    ratio_df = DataFrame(shelfcounts_yr).fillna(0).set_index('book_title').T
    # for each shelf, calculate the proportion of total books that were placed on it
    # make dataframe of the number of non-zero entries for each shelf
    # total columns minus number of columns with value zero
    nonzero_df = (len(ratio_df.columns) - (ratio_df == 0).astype(float).sum(axis=1))/len(ratio_df.columns)
    ratio_dict = nonzero_df.to_dict()
    sel_shelfcounts = []
    for book in shelfcounts_yr:
        bookdict = {}
        bookdict['book_title'] = book['book_title']
        del book['book_title']
        # only include shelf if below a certain ratio of books use that shelf
        for shelf, score in book.iteritems():
            ratio = ratio_dict[shelf]
            # ignore shelves with too many and too few uses
            # on the less than side, gets rid of shelves like 'to-read', 'ebooks', 'to-buy', 'wish-list', '2014', '2013', 'own', 'owned', 'owned-books', 'ebook', '2015', 'library', '2016', 'library', 'wishlist', 'kindle', 'fiction', 'currently-reading', 'default', 'favorites', 'books-i-own' but not 'history' 'non-fiction' 'series' etc.
            if ratio < max_ratio and ratio > (float(min_books_using_shelf)/len(ratio_df.columns)):
                bookdict[shelf] = score
            else:
#                print 'denied ' + shelf + ' with ratio ' + str(ratio)
                continue
        sel_shelfcounts.append(bookdict)

    # trim list of shelves to only the top n shelves for each book
    sel2_shelfcounts = []
    for book in sel_shelfcounts:
        bookdict = {}
        bookdict['book_title'] = book['book_title']
        del book['book_title']
        # gets highest n scores - 10 because that's how many show up on main book page on Goodreads
        for shelf, score in nsmallest(n, book.iteritems(), key=lambda (k, v): (-v, k)):
            bookdict[shelf] = score
        sel2_shelfcounts.append(bookdict)

    # do the same for the raw counts
    raw_sel2_shelfcounts = []
    for book in raw_shelfcounts_yr:
        bookdict = {}
        bookdict['book_title'] = book['book_title']
        del book['book_title']
        # gets highest n scores - 10 because that's how many show up on main book page on Goodreads
        for shelf, score in nsmallest(n, book.iteritems(), key=lambda (k, v): (-v, k)):
            bookdict[shelf] = score
        raw_sel2_shelfcounts.append(bookdict)

    return DataFrame(sel2_shelfcounts).fillna(0).set_index('book_title'), \
           DataFrame(raw_sel2_shelfcounts).fillna(0).set_index('book_title')


def visualize_standardization_methods(df):
    print 'visualizing standardization methods'
    # standardize data
    std_scale = preprocessing.StandardScaler().fit(df)
    df_std = std_scale.transform(df)

    minmax_scale = preprocessing.MinMaxScaler().fit(df)
    df_minmax = minmax_scale.transform(df)

    plt.figure(figsize=(8, 6))

    try:
        testx = 'wish-list'
        testy = 'young-adult'
        plt.scatter(df[testx], df[testy],
                    color='green', label='input scale', alpha=0.5)
    except:
        print 'shelves not available for scatterplot'

    plt.scatter(df_std[:, 0], df_std[:, 1], color='red',
                label='Standardized', alpha=0.3)

    plt.scatter(df_minmax[:, 0], df_minmax[:, 1],
                color='blue', label='min-max scaled [min=0, max=1]', alpha=0.3)

    plt.title('Plot of example shelves')
    plt.xlabel(testx)
    plt.ylabel(testy)
    plt.legend(loc='upper left')
    plt.grid()
    plt.tight_layout()
    plt.show()


def get_genres_from_database(book_title_list):
    '''
    looks in database for book_title and grabs associated genre

    need to account for conflicting genres (books in 2 different genres)
    '''
    print 'getting genres from database'
    databasefile = "diss.sqlite"

    sf = ["Science Fiction", "Science fiction", "science fiction", "science-fiction", "SF", "sf", "Fantasy/Sci-fi", "Science Fiction & Fantasy", "Science fiction & fantasy", "scifi", "sci fi", "sci-fi"]
    horror = ["Horror", "horror"]
    romance = ["Romance", "romance", "romances"]
    detective = ["Mystery", "Mysteries", "Detective Fiction", "noir", "Thrillers", "Mystery & Crime", "Mystery & Suspense", "mystery"]
    biography = ["Biography", "biography & autobiography", "Memoir", "Biographies", "Biography & Memoir", "biography"]
    history = ["History", "history", "histories"]
    genres = [sf, horror, romance, detective, biography, history]

    conn = sqlite3.connect(databasefile)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("select genre, site, book_title "
                "from Reviews "
                "where genre <> '' and site = 'goodreadsreviews'")  # genre not empty/null
                # can add LIMIT 5 to do a subset of reviews
    database = [dict(row) for row in cur]
    # database is a list of dicts where {'body': 'text', 'genre': 'genre', 'site': 'site'}
    cur.close()
    conn.close()

    book_genres = []
    for book_title in book_title_list:
        potential_genres = []
        for review in database:
            if review['book_title'] == book_title:
                potential_genres.append(review['genre'])

        book_genre = float('NaN')
        for potential_genre in potential_genres:
            for genre in genres:
                if potential_genre in genre:
                    book_genre = genre[0]
                    break
            if book_genre:
                break

        book_genres.append(book_genre)

    return book_genres


def save_cluster_model(df, n=5):
    km = KMeans(n_clusters=n)
    km.fit(df)
    joblib.dump(km, 'cluster_model.pkl')


def top_terms_per_cluster(df, terms, num_clusters=5, n=10):
    '''
    finds the top n words that are nearest to the cluster centroid
    '''
    top_terms = {}
    titles = {}

    print "Top terms per cluster:"
    # sort cluster centers by proximity to centroid
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]

    for i in range(num_clusters):
        finals = []
        print "Cluster %d words:" % i
        for ind in order_centroids[i, :n]:
            print ' %s, ' % terms[ind],
            finals.append(terms[ind])
        top_terms[i] = finals
        print ''
        print ''

        books = []
        print "Cluster %d titles:" % i
        try:
            for title in df.ix[i]['book_title'].values.tolist():
                print ' %s,' % title,
                books.append(title)
        except:
            temp = df.loc[resultsdf['cluster'] == 1]
            print temp['book_title'].iloc[0]
            books.append(temp['book_title'].iloc[0])
        titles[i] = books
        print ''
        print ''

    return top_terms, titles


def visualize_clusters_kmeans(df, resultsdf, clusters, top_terms):
    print 'visualizing clusters (kmeans)'
    # make matrix showing the similarity between each document and every other
    # document in the corpus
    dist = 1 - cosine_similarity(df.T)

    # clean up top terms to remove repeats
    all_terms = []
    for k, v in top_terms.iteritems():
        all_terms.extend(v)
    unique_terms = [k for k, n in Counter(all_terms).iteritems() if n == 1]
    top_unique_terms = {}
    for k, v in top_terms.iteritems():
        top_unique_terms[k] = []
        for term in v:
            if term in unique_terms:
                top_unique_terms[k].append(term)
    # turn lists of terms into a string
    for k, v in top_unique_terms.iteritems():
        top_unique_terms[k] = ', '.join(v)

    MDS()
    # convert two components as we're plotting points in a two-dimensional plane
    # "precomputed" because we provide a distance matrix
    # we will also specify 'random_state' so the plot is reproducible.
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
    pos = mds.fit_transform(dist)  # shape (n_components, n_samples)
    xs, ys = pos[:, 0], pos[:, 1]

    # set up colors for clusters using a dict
    cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a',
                      4: '#66a61e', 5: '#d3d3d3', 6: '#43a2ca', 7: '#a8ddb5',
                      8: '#e34a33', 9: '#fdbb84', 10: '#fee8c8', 11: '#8856a7',
                      12: '#9ebcda', 13: '#e0ecf4', 14: '#e6550d', 15: '#fdae6b',
                      16: '#c994c7', 17: '#fa9fb5', 18: '#993404', 19: '#fdbe85'}

    cluster_names = top_unique_terms

    vizdf = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=resultsdf['book_title']))

    # group by cluster
    groups = vizdf.groupby('label')

    # set up plot
    fig, ax = plt.subplots(figsize=(17, 9))  # set size
    ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling

    # iterate through groups to layer the plot
    # note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=12,
                label=cluster_names[name], color=cluster_colors[name],
                mec='none')
        ax.set_aspect('auto')
        ax.tick_params(
            axis='x',         # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelbottom='off')
        ax.tick_params(
            axis='y',         # changes apply to the y-axis
            which='both',      # both major and minor ticks are affected
            left='off',        # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelleft='off')

    ax.legend(numpoints=1)  # show legend with only 1 point

    # add label in x,y position with the label as the film title
    for i in range(len(vizdf)):
        ax.text(vizdf.iloc[i]['x'], vizdf.iloc[i]['y'], vizdf.iloc[i]['title'], size=8)

    plt.show()

    plt.close()


def visualize_clusters_hierarchical(df, distances, x, y, names, method='ward'):
    '''
    distances is a distance matrix
    https://de.dariah.eu/tatom/working_with_text.html
    '''
    print 'visualizing clusters (hierarchical), method: ' + method

    if method == 'hdbscan':
        clusterer = hdbscan.HDBSCAN(metric='precomputed', min_cluster_size=5,
                                    min_samples=5)  # higher = more outliers
        clusterer.fit(distances)
        color_palette = sns.color_palette('deep', (clusterer.labels_.max()+1))
        cluster_colors = [color_palette[z] if z >= 0
                          else (0.5, 0.5, 0.5)
                          for z in clusterer.labels_]
        cluster_member_colors = [sns.desaturate(n, p) for n, p in
                                 zip(cluster_colors, clusterer.probabilities_)]
        plt.scatter(x, y, s=30, linewidth=0, c=cluster_member_colors, alpha=0.25)
        plt.show()
        plt.close()

        linkage_matrix = clusterer.condensed_tree_.to_pandas()
        clusterer.condensed_tree_.plot()

        for i in range(clusterer.labels_.max()+1):
            print 'Cluster ' + str(i) + ':'
            for f, num in enumerate(clusterer.labels_):
                if i == num:
                    print df.index.values[f],
            print ''
            print ''

    elif method in ['ward', 'single', 'average', 'weighted', 'centroid']:
        # make matrix showing the similarity between each document and every other
        # document in the corpus
        linkage_matrix = linkage(distances, method)

        # adjust height of dendrogram based on how many shelves there are
        print linkage_matrix.shape
        num_rows, num_columns = linkage_matrix.shape
        height = num_rows * 0.185
        fig, ax = plt.subplots(figsize=(15, height))

        plt.tick_params(
            axis='x',         # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelbottom='off')

        plt.tight_layout()

        plt.savefig('hierarchical_clusters.png', dpi=200)
        print 'saved hierarchical_clusters.png'
        plt.close()

        # show a scatterplot of the clustering results
        # color labels of interest red
        colors = []
        for i, shelf in enumerate(df.index.values):
            # enter shelves of interest here to color them red
            if shelf in ['wwi', 'ww-i', 'world-war-i', 'world-war-1', 'ww1']:
                colors.append('red')
            else:
                colors.append('black')

        plt.scatter(x, y, c=colors, s=5)  # alternative: c=fc


def show_silhouette_charts(df, clustering_method, num_cluster_list):
    print 'showing silhouette charts'
    X = 1 - cosine_similarity(df)

    plt.hist(X, normed=True, bins=30)
    plt.ylabel('Probability')

    for n_clusters in num_cluster_list:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        if clustering_method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=10)
            cluster_labels = clusterer.fit_predict(X)
        elif clustering_method == 'ward':
            clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
            cluster_labels = clusterer.fit_predict(X)
        elif clustering_method == 'meanshift':
            bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)
            clusterer = MeanShift(bandwidth=bandwidth, bin_seeding=True)
            cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the
        # formed clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print "For n_clusters = " + str(n_clusters) + " the average silhouette_score is: " + str(silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        # 2nd plot showing the actual clusters formed
        colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors)

        has_cluster_centers = ['kmeans', 'meanshift']
        if clustering_method in has_cluster_centers:
            # Labeling the clusters
            centers = clusterer.cluster_centers_
            # Draw white circles at cluster centers
            ax2.scatter(centers[:, 0], centers[:, 1],
                        marker='o', c='white', alpha=1, s=200)

            for i, c in enumerate(centers):
                ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50)

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(("Silhouette analysis for clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')

        plt.show()


def get_top_shelves(data):
    print 'getting top shelves'
    shelflist = []
    countlist = []

    for item in data:
        for key, value in item.iteritems():
            if key == 'shelf':
                for shelfname in value:
                    shelflist.append(shelfname)
            if key == 'people':
                for peoplecount in value:
                    newpeoplecount = peoplecount.replace(',', '')
                    countlist.append(newpeoplecount[:-7])

    result = {}

    for i in range(0, len(shelflist)):
        if shelflist[i] not in result:
            result[shelflist[i]] = int(countlist[i])
        else:
            result[shelflist[i]] += int(countlist[i])

    sorted_result = sorted(result.items(), key=operator.itemgetter(1), reverse=True)
    df = DataFrame(sorted_result)
    return df


def reduce_dimensionality(df, distances, dim_type):
    print 'reducing dimensionality'
    if dim_type == 'mds':
        # convert two components as we're plotting points in a two-dimensional plane
        # "precomputed" because we provide a distance matrix
        # we will also specify 'random_state' so the plot is reproducible.
        mds = manifold.MDS(n_components=2, dissimilarity="precomputed", random_state=1)
        pos = mds.fit_transform(distances)  # shape (n_components, n_samples)
    if dim_type == 'isomap':
        pos = manifold.Isomap(n_neighbors=10, n_components=2).fit_transform(distances)
    if dim_type == 'pca':
        pca = decomposition.PCA(n_components=2, random_state=1)
        pos = pca.fit_transform(distances)
    if dim_type == 'sparsepca':
        pca = decomposition.SparsePCA(n_components=2, random_state=1)
        pos = pca.fit_transform(distances)
    if dim_type == 'tsne':
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=1)
        pos = tsne.fit_transform(distances)
    if dim_type == 'spectral':
        se = manifold.SpectralEmbedding(n_components=2, n_neighbors=10, random_state=1)
        pos = se.fit_transform(distances)
    if dim_type == 'lle':
        methods = ['standard', 'ltsa', 'hessian', 'modified']
        lle = manifold.LocallyLinearEmbedding(n_neighbors=10, n_components=2, method=methods[1], random_state=1)
        pos = lle.fit_transform(distances)
    if dim_type == 'trunc':
        tsvd = decomposition.TruncatedSVD(n_components=2, random_state=1)
        pos = tsvd.fit_transform(distances)

    xs, ys = pos[:, 0], pos[:, 1]
    names = df.index.values

    for x, y, name in zip(xs, ys, names):
        plt.scatter(x, y, s=5)
#        plt.text(x, y, name)  # can be overwhelming with lots of data points
    plt.show()
    return xs, ys, names


def visualize_data_histogram(x, y):
    print 'visualizing data histogram'
    # make matrix of pairwise distances
    vects = np.column_stack((x, y))
    dists = pairwise_distances(vects)

    # make histogram of pairwise distances
    plt.hist(dists, bins=40)


def visualize_data_heatmap(df):
    print 'visualizing data heatmap'
    # make heatmap of dataframe
    fig, ax = plt.subplots()
    ax.pcolor(df)
    plt.yticks(np.arange(0.5, len(df.index), 1), df.index)
    plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns)
    fig.set_size_inches(10, 10)
    plt.savefig("heatmap.png")


def get_streamgraph_data(df, shelves):
#    segment = df.loc[:, shelves]
    for shelf in shelves:
        print shelf, str(int(df[shelf].sum()))


if __name__ == '__main__':
    # get data from Goodreads website scrape
    df, df_raw = get_shelf_data(shelf_json='goodreadsshelves_sample.json',
                                max_ratio=0.20, min_books_using_shelf=25)

    # get data from specific year(s) via the Internet Archive Wayback Machine
#    years, max_ratio, min_books, n = [2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017], .45, 0, 10
#    df, df_raw = get_wayback_shelf_data(shelf_json='E:/0-docs/diss/data/goodreadsshelves-random_wayback.json',
#                                        max_ratio=max_ratio, min_books_using_shelf=min_books, n=n, years=years)
#    print 'number of results for ' + str(years) + ': (unique books, shelves)'
#    print df_raw.shape
#    print 'total shelvings in dataset for ' + str(years) + ':'
#    print df_raw.values.sum()

    # visualize the effect of vairous standardization methods to determine
    # if you should standardize your data
    visualize_standardization_methods(df)

    # create distance/proximity matrix
    dist = 1 - cosine_similarity(df.T)

    # perform dimensionality reduction
    x, y, names = reduce_dimensionality(df.T, dist, dim_type='tsne')

    # cluster data
    clustering_type = 'hierarchical'
    if clustering_type == 'kmeans':
        # run with 'None' to find best number of clusters to use, then update with that number and run again
        num_clusters = None

        if not num_clusters:
            # preliminary data visualization
            visualize_data_heatmap(df)

            # decide how many clusters to use
            show_silhouette_charts(df, clustering_method='ward', num_cluster_list=[2, 3, 6, 15])
            # afterwards, update num_clusters with best number and run again
        if num_clusters:
            # comment this out once it's saved
            save_cluster_model(df, num_clusters)

            km = joblib.load('cluster_model.pkl')
            clusters = km.labels_.tolist()

            book_titles = list(df.index.values)
            results = {'book_title': book_titles, 'cluster': clusters}
            resultsdf = DataFrame(results, index=[clusters], columns=['book_title', 'cluster'])

            # get traditional genre labels if available - do this after clustering so you don't bias the model
            genre_column = np.asarray(get_genres_from_database(resultsdf['book_title'].tolist()))
            resultsdf['book_genre'] = genre_column

            # do k-means clustering
            terms = list(df)
            top_terms, titles = top_terms_per_cluster(resultsdf, terms, num_clusters, 10)
            visualize_clusters_kmeans(df, resultsdf, clusters, top_terms)
    elif clustering_type == 'hierarchical':
        # instead of using original dataframe, use reduced data
        reduced_df = pd.DataFrame({'x': x.tolist(), 'y': y.tolist()}, index=names)
        visualize_clusters_hierarchical(reduced_df, dist, x, y, names, method='ward')
