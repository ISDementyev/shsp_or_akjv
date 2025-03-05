import re, nltk
import string
from nltk import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, words


def read_corpus(fname):
    """
    Reads corpus from a file containing it.
    :param fname: Name of the file (str)
    :return: File contents (corpus) (str)
    """

    with open(fname) as f:
        return f.read()

def process_string(string_):
    """
    Stems string and removes stopwords
    :param string_: String to process (str)
    :return: Processed string (list)
    """
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words("english")

    # remove digits
    string_ = re.sub(r"\d+", "", string_)

    # tokenize remaining string
    tokens = word_tokenize(string_)

    words_clean = []
    for word in tokens:
        if (word not in stopwords_english) and (word not in string.punctuation):
            stem_word = stemmer.stem(word)
            words_clean.append(stem_word)

    return words_clean

if __name__ == "__main__":
    nltk.download("punkt_tab")
    nltk.download("stopwords")
    akjv_file = "/Users/ilya/Desktop/DL_AI/shsp_or_kjb/kjb_corpus/AKJV.txt"
    akjv_corpus = read_corpus(akjv_file)
    tokenized_akjv = word_tokenize(akjv_corpus)
    processed_words_akjv = process_string(akjv_corpus)
    print(len(tokenized_akjv))
    print(len(processed_words_akjv)) # reduced by 63%
