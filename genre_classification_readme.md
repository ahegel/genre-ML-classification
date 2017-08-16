# Genre Classification
Using machine learning to classify Goodreads book reviews based on genre, revealing the associated vocabulary of user reviews for each genre and trends in word use over time.

## Abstract
How predictable are readers' responses to books in a given genre? This project uses machine learning to classify book reviews into genres and then compares the predictability of each genre, ultimately showing which genres have a well-defined vocabulary for reader responses and which genres inspire a more diverse range of responses.

## Get book review data
To start, we need book reviews. In [`genre_classification.py`](https://github.com/ahegel/machine-learning-genre/blob/master/genre_classification.py), the functions `get_reviews()` and `get_reviews_by_year()` get book review data from a SQLite database, but you can substitute your own function based on your project's data structure.

The data retrieval function is customizable to get data from different corpora (Goodreads, professional reviews like the Los Angeles Review of Books, or academic articles), from different genres or Goodreads shelves, from individual years or groups of years (to compare classification accuracy over time), and from different pages of search results. 

For example, if you run the program to extract book reviews from Goodreads where the reviewer placed the book on the shelf "creepy" [Goodreads "creepy" shelf](https://www.goodreads.com/shelf/show/creepy), the resulting DataFrame has this structure:

body | book_title | book_author | review_pub_date | reviewer_bookshelves | shelf_genre
--- | --- | --- | --- | --- | ---
Solid 4.5. Although the second part of the boo... | Le Rituel | Adam Nevill | 2016-02-24 | [creepy] | On shelves
This is a delightfully chilly book! This is a ... | The Graveyard Book | Neil Gaiman | 2013-04-13 | [creepy, cute, ghosts, science-fiction, thriller] | On shelves
I saved the end for this morning, and wept. Th... | The House of Mirth | Edith Wharton | 2009-09-29 | [audible, 2015-11, 0-kindle, books-read-in-2015] | Not on shelves
If you've seen the movie it's probably best to... | World War Z: An Oral History of the Zombie War | Max Brooks | 2014-12-18 | [] | Not on shelves

The size of the DataFrame is 1332 rows (each corresponding to one review) and 9 columns of metadata. Half of the results are books on the creepy shelf, and half are a random selection of books not on that shelf, allowing us to train a classifier to recognize creepy books in the next step.

## Train a model to classify the genre of reviews
Using the `make_predictions()` function, we can determine how difficult it is to train a machine learning model to distinguish reviews of creepy books from reviews of other books. The function's results include a table showing the overall scores for the classifier, as well as the list of most predictive words:

| | precision | recall | f1-score | support
|--- | --- | --- | --- | ---
|On creepy shelf | 0.69 | 0.64 | 0.67 | 666
|Not on creepy shelf | 0.67 | 0.71 | 0.69 | 666
|avg / total | 0.68 | 0.68 | 0.68 | 1332

measure | description
--- | ---
precision | true positives / (true positives + false positives)
recall | true positives / (true positives + false negatives)
f1-score | weighted mean of precision and recall
support | number of occurrences of each class

Words that predict a review is about a creepy book:

score | word
--- | ---
1.54145273279 | ghost
1.44308621722 | disturbing
1.34830032862 | gothic
1.28083000982 | stories
1.17869574451 | inside
1.16994647823 | dead
1.16331482283 | house
1.16276342299 | scary
1.14944865678 | twist
1.13319235119 | type
1.12720247093 | pretty
1.11549824486 | horror
1.11444891207 | super
1.09978202644 | weird
1.07459981228 | mama
1.02994779539 | ghosts
1.01634480127 | lately
1.00726875706 | chilling
1.00059297401 | strange
0.994034481582 | lines

Words that predict a review is NOT about a creepy book:

score | word
--- | ---
-1.56235397666 | time
-1.15093084301 | many
-1.14776953079 | play
-1.14235528659 | seemed
-1.13073754049 | life
-1.06215322227 | care
-1.05125044861 | history
-1.0302060619 | sweet
-0.981626217684 | secondary
-0.970966795754 | emotionally
-0.953212212225 | recommended
-0.952656736519 | tried
-0.936802237232 | later
-0.92617967251 | thing
-0.921969123412 | vampire
-0.908862331791 | due
-0.895132645555 | chance
-0.891780176589 | middle
-0.876579692027 | change
-0.872261901834 | important

We can compare how these top words change over time in reviews, tracking which words characterize creepy books at different points in time:

![Top most predictive words for creepy books](/images/classification_creepy_topwords.png)

## Comparing classification results between genres

![Classification boxplot](/images/classification_boxplot.png)

## Comparing classification results over time

![Predictability of creepy reviews on Goodreads](/images/classification_creepy.png)

![Classification over time comparison by platform](/images/classification_overtime.jpg)
