import re, nltk, json, math, random, string
from nltk import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def read_corpus(fname):
    """
    Reads corpus from a file containing it.
    :param fname: Name of the file (str)
    :return: File contents (corpus) (str)
    """

    with open(fname) as f:
        return f.read()

def shsp_dict():
    """
    Returns dict of Shakespeare phonetic and stem text
    e.g. one of the entries is dict["ETWRT"] = "edward"
    :return: The aforementioned dict
    """
    fname = "/Users/ilya/Desktop/DL_AI/shsp_or_kjb/shsp/WordForms.txt"
    with open(fname) as r:
        lines = r.readlines()

    shsp_dict = dict()
    for line in lines[1:]:
        wordform_id, plaintext, phonetictext, stemtext, occurences = line.split(",")
        phonetictext = phonetictext.lower().replace("~", "")
        stemtext = stemtext.lower().replace("~", "")
        plaintext = plaintext.lower().replace("~", "")
        shsp_dict[phonetictext.lower()] = plaintext
        shsp_dict[stemtext] = plaintext

    # print(shsp_dict)
    return shsp_dict

def process_string(string_, kjv=False):
    """
    Stems string and removes stopwords
    :param string_: String to process (str)
    :return: Processed string (list)
    """
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words("english")
    label = int(kjv)
    if not label: # if label points to shakespeare
        shakespeare_dict = shsp_dict()
        string_ = string_.replace("[p]", "")
        string_list = string_.split(" ")
        # print(string_list[:50])

        # for i in range(len(string_list)):
        #     if string_list[i] in shakespeare_dict:
        #         string_list[i] = shakespeare_dict[string_list[i]]
        # string_ = " ".join(string_list)

    # remove digits
    string_ = re.sub(r"\d+", "", string_)

    # tokenize remaining string
    tokens = word_tokenize(string_)

    words_clean = []
    punctuation = string.punctuation

    for word in tokens:
        if ((word not in stopwords_english)
                and (word not in punctuation)
                and ("~" not in word)):
            stem_word = stemmer.stem(word)
            # if stem_word in shakespeare_dict and not label:
            #     stem_word = shakespeare_dict[stem_word]
            words_clean.append(stem_word)

    if not label:
        for i in range(len(words_clean)):
            if words_clean[i] in shakespeare_dict:
                words_clean[i] = shakespeare_dict[words_clean[i]]

        tokenize_final = word_tokenize(" ".join(words_clean))


        return [word for word in tokenize_final if word not in ["'d", "'t", "'s"]]#words_clean

    else:
        return words_clean

def count_freqs(freqs, word_list, kjv=True):
    """
    Counts the frequency of the word
    :param freqs: Frequency dict (dict)
    :param word_list: The list of words to count
    :return: [(Word, label): frequency] key-value pair
    """
    label = int(kjv)

    for word in word_list:
        pair = (word, label)
        if pair not in freqs:
            freqs[(word, label)] = 1
        else:
            freqs[(word, label)] += 1

    return freqs

def write_json_file(fname, obj_):
    """
    Writes an iterable to the hard drive as a JSON file
    :param fname: What to call your file name (str)
    :param obj_: The object to be saved to hard drive (any Python iterable class)
    :return: None
    """
    with open(fname, "w") as w:
        json.dump(obj_, w)

def train_naive_bayes(freqs, train_x, train_y):
    """
    Train the Naive Bayes model
    :param freqs: dictionary from word, label
    :param train_x: a list of words
    :param train_y: a list of labels corresponding to the words (0 for Shakespeare, 1 for AKJV)
    :return: Log Prior and Log Likelihood Scores (tuple: (float, dict[str] = float))
    """
    loglikelihood = dict()
    logprior = 0

    # calculate number of unique words in vocab
    vocab = set([word for word, label in freqs.keys()])
    V = len(vocab)

    # constants needed for Laplacian smoothing and Bayes implementation
    N_shsp = N_akjv = V_shsp = V_akjv = 0
    for pair in freqs.keys():
        if pair[1] == 0:
            N_shsp += freqs[pair]
            V_shsp += 1
        elif pair[1] == 1:
            N_akjv += freqs[pair]
            V_akjv += 1

    D = len(train_y)
    D_shsp = len([y for y in train_y if y == 0])
    D_akjv = len([y for y in train_y if y == 1])

    logprior = math.log(D_akjv) - math.log(D_shsp)

    for word in vocab:
        freq_akjv = freqs.get((word, 1), 0)
        freq_shsp = freqs.get((word, 0), 0)

        # calculate probability that a word is from a Shakespeare work or the AKJV bible
        p_w_akjv = (freq_akjv + 1) / (N_akjv + V)
        p_w_shsp = (freq_shsp + 1) / (N_shsp + V)

        # calculate log likelihood of the word
        loglikelihood[word] = math.log(p_w_akjv / p_w_shsp)

    return logprior, loglikelihood

def naive_bayes_predict(sentence, logprior, loglikelihood):
    proc_sentence = process_string(sentence)
    probability = 0
    probability += logprior

    for word in proc_sentence:
        if word in loglikelihood:
            probability += loglikelihood[word]

    return probability

def test_naive_bayes(test_x, test_y, logprior, loglikelihood, naive_bayes_predict=naive_bayes_predict):
    y_hats = []
    print(f"len of test_x: {len(test_x)}")
    for word in test_x:
        # print(word)
        if naive_bayes_predict(word, logprior, loglikelihood) > 0:
            y_hat_i = 1
        else:
            y_hat_i = 0

        y_hats.append(y_hat_i)

    abs_vals = [abs(yh - yt) for yh, yt in zip(y_hats, test_y)]
    error = sum(abs_vals) / len(abs_vals)
    accuracy = 1 - error

    return accuracy

def balance_corpus(words, target_reduction_percent=21):
    """
    Randomly removes a specified percentage of words from the Shakespeare corpus.

    Args:
        shakespeare_words (list): List of words from Shakespeare's works
        target_reduction_percent (float): Percentage of words to remove (default: 21)

    Returns:
        list: Reduced Shakespeare corpus with approximately 21% fewer words
    """
    # Calculate how many words to keep
    words_to_keep = int(len(words) * (target_reduction_percent) / 100)

    # Randomly sample the required number of words
    balanced_corpus = random.sample(words, words_to_keep)

    return balanced_corpus

if __name__ == "__main__":
    nltk.download("punkt_tab")
    nltk.download("stopwords")

    akjv_file = "/Users/ilya/Desktop/DL_AI/shsp_or_kjb/kjb_corpus/AKJV.txt"
    shsp_dict_file = "/Users/ilya/Desktop/DL_AI/shsp_or_kjb/shsp/WordForms.txt"
    shsp_file = "/Users/ilya/Desktop/DL_AI/shsp_or_kjb/shsp/Paragraphs.txt"

    # akjv_corpus = read_corpus(akjv_file)
    # tokenized_akjv = word_tokenize(akjv_corpus)
    # processed_words_akjv = process_string(akjv_corpus)
    #
    # print(len(tokenized_akjv))
    # print(len(processed_words_akjv)) # reduced by 63%
    shakespeare_corpus = read_corpus(shsp_file)
    processed_shakespeare = process_string(shakespeare_corpus)
    write_json_file("processed_shakespeare.json", processed_shakespeare)
    print(processed_shakespeare[:100])



