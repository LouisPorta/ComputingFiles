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
    dataset = pd.read_csv(csv_file_path)
    all_text = " ".join(np.unique(dataset.question))
    sentences = [word_tokenize(sent) for sent in sent_tokenize(all_text)]

    # POS tag using Universal tagset
    tagged_sentences = pos_tag_sents(sentences, tagset='universal')


    adjectives = [
        word.lower()
        for sent in tagged_sentences
        for word, tag in sent
        if tag == 'ADJ'
    ]

    # Count
    wordcounter = collections.Counter(adjectives)
    return(wordcounter.most_common(5))

# Task 2: Top N 3-grams and average trigram length (stemmed and raw)
def topN_3grams(csv_file_path, N):
    """
    Example:
        >>> topN_3grams('train.csv', 5)
        Returns the top-N (eg. top-5) most frequent 3-grams (trigrams) of stemmed and non-stemmed tokens along with their normalized frequency from the questions column in a given CSV file
        Expected Output: ([(('what', 'is', 'the'), 0.0105), (('what', 'is', 'a'), 0.0053), (('what', 'are', 'the'), 0.0039), (('when', 'wa', 'the'), 0.0016), (('what', 'wa', 'the'), 0.0015)], [(('what', 'is', 'the'), 0.0105), (('what', 'is', 'a'), 0.0053), (('what', 'are', 'the'), 0.0039), (('when', 'was', 'the'), 0.0016), (('what', 'was', 'the'), 0.0015)], 2.9333, 3.0667)
        Note: Last 2 values i.e. 2.9333 and 3.0667 are the average trigram length for the stemmed and raw.
    """
    dataset = pd.read_csv(csv_file_path)
    all_text = " ".join(np.unique(dataset.question))
    sentences = [word_tokenize(sent) for sent in sent_tokenize(all_text)]

    stm = nltk.stem.PorterStemmer()
    norm_sentences = [[words.lower() for words in sent] for sent in sentences]
    stemmed = [[stm.stem(words) for words in sent] for sent in norm_sentences]

    ngrams = [list(nltk.ngrams(sent, 3)) for sent in norm_sentences]
    ngrams_counter = collections.Counter(gram for sent in ngrams for gram in sent)
    ngrams_freq = [(ng, round(freq/sum([len(sent) for sent in ngrams]),4)) for ng, freq in ngrams_counter.most_common(N)]
    ngram_avg = len([letters for ng, freq in ngrams_freq for words in ng for letters in words])/len([words for ng, freq in ngrams_freq for words in ng])

    st_ngrams = [list(nltk.ngrams(sent, 3)) for sent in stemmed]
    st_ngrams_counter = collections.Counter(gram for sent in st_ngrams for gram in sent)
    st_ngrams_freq = [(ng, round(freq/sum([len(sent) for sent in ngrams]),4)) for ng, freq in st_ngrams_counter.most_common(N)]
    st_ngram_avg = len([letters for ng, freq in st_ngrams_freq for words in ng for letters in words])/len([words for ng, freq in st_ngrams_freq for words in ng])

    
    return [st_ngrams_freq, ngrams_freq,round(st_ngram_avg,4),round(ngram_avg,4)]


    

# Task 3: Predict Answer Sentences based on TF-IDF
def sim_tfidf(csv_file_path):
    """
    Example:
    >>> sim_tfidf('train.csv')
    Returns the proportion of questions can be accurately answered using the tf.idf feature
    Expected Output: 0.119

    """
    dataset = pd.read_csv(csv_file_path)
    unique_questions = dataset.question.unique()
    all_answers = dataset.answer
    corpus = np.concatenate([unique_questions, all_answers])

    vectorizer = TfidfVectorizer(stop_words='english')
    vectorizer.fit(corpus)

    correct = 0
    total = 0

    for question in unique_questions:
      df = dataset[dataset.question == question]
      answers = df.answer.tolist()
      labels = df.label.tolist()

      tfidf = vectorizer.transform([question] + answers)
      best_index = cosine_similarity(tfidf[0], tfidf[1:]).argmax()

      correct += labels[best_index]
      total += 1

    return correct / total

    
# DO NOT MODIFY BELOW
if __name__ == "__main__":

    import doctest
    doctest.testmod(optionflags=doctest.ELLIPSIS)
    file_path = 'train.csv'
    print("Top 5 ADJs:", topN_pos_adj(file_path, 5))
    print("\nTop 5 Trigrams (Stemmed & Raw):", topN_3grams(file_path, 5))
    print("\nProportion of questions that can be accurately answered based on TF-IDF:", 
      str(sim_tfidf(file_path)).replace("\n", ""))
