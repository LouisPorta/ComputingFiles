import nltk
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('universal_tagset')

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag_sents
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import collections
import pandas as pd

# Task 1: Top N ADJECTIVES in questions
def topN_pos_adj(csv_file_path, N):
    """
    Example:
    >>> topN_pos('train.csv', 5)
    Returns the top-N (eg. top-5) common ADJECTIVE in the questions
    Expected Output: [('many', 140), ('first', 38), ('much', 29), ('old', 25), ('new', 21)]

    """

# Task 2: Top N 3-grams and average trigram length (stemmed and raw)
def topN_3grams(csv_file_path, N):
    """
    Example:
        >>> topN_3grams('train.csv', 5)
        Returns the top-N (eg. top-5) most frequent 3-grams (trigrams) of stemmed and non-stemmed tokens along with their normalized frequency from the questions column in a given CSV file
        Expected Output: ([(('what', 'is', 'the'), 0.0105), (('what', 'is', 'a'), 0.0053), (('what', 'are', 'the'), 0.0039), (('when', 'wa', 'the'), 0.0016), (('what', 'wa', 'the'), 0.0015)], [(('what', 'is', 'the'), 0.0105), (('what', 'is', 'a'), 0.0053), (('what', 'are', 'the'), 0.0039), (('when', 'was', 'the'), 0.0016), (('what', 'was', 'the'), 0.0015)], 2.9333, 3.0667)
        Note: Last 2 values i.e. 2.9333 and 3.0667 are the average trigram length for the stemmed and raw.
    """

# Task 3: Predict Answer Sentences based on TF-IDF
def sim_tfidf(csv_file_path):
    """
    Example:
    >>> sim_tfidf('train.csv')
    Returns the proportion of questions can be accurately answered using the tf.idf feature
    Expected Output: 0.119

    """
    
# DO NOT MODIFY BELOW
if __name__ == "__main__":

    import doctest
    doctest.testmod(optionflags=doctest.ELLIPSIS)
    file_path = 'train.csv'
    print("Top 5 ADJs:", topN_pos_adj(file_path, 5))
    print("\nTop 5 Trigrams (Stemmed & Raw):", topN_3grams(file_path, 5))
    print("\nProportion of questions that can be accurately answered based on TF-IDF:", 
      str(sim_tfidf(file_path)).replace("\n", ""))
