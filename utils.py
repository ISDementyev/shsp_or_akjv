import nltk
from nltk.tokenize import word_tokenize


def read_corpus(fname):
    """
    Reads corpus from a file containing it.
    :param fname: Name of the file (str)
    :return: File contents (corpus) (str)
    """

    with open(fname) as f:
        return f.read()

if __name__ == "__main__":
    nltk.download("punkt_tab")
    akjv_file = "/Users/ilya/Desktop/DL_AI/shsp_or_kjb/kjb_corpus/AKJV.txt"
    akjv_corpus = read_corpus(akjv_file)
    tokenized_akjv = word_tokenize(akjv_corpus)
    print(type(tokenized_akjv))