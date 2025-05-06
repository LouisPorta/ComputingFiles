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

import unittest
import a3_1


class TestBasic(unittest.TestCase):
    def test_q1(self):
        result = a3_1.topN_pos_adj('train.csv', 5)
        self.assertTrue(result == [('many', 140), ('first', 38), ('much', 29), ('old', 25), ('new', 21)])

    def test_q2(self):
        result = a3_1.topN_3grams('train.csv', 5)
        self.assertTrue(result == ([(('what', 'is', 'the'), 0.0105), (('what', 'is', 'a'), 0.0053),
                                     (('what', 'are', 'the'), 0.0039), (('when', 'wa', 'the'), 0.0016),
                                     (('what', 'wa', 'the'), 0.0015)],
                                    [(('what', 'is', 'the'), 0.0105), (('what', 'is', 'a'), 0.0053),
                                     (('what', 'are', 'the'), 0.0039), (('when', 'was', 'the'), 0.0016),
                                     (('what', 'was', 'the'), 0.0015)],
                                    2.9333,
                                    3.0667))

    def test_q3(self):
        result = a3_1.sim_tfidf('train.csv')
        self.assertEqual(result, 0.119)


if __name__ == "__main__":
    unittest.main()
