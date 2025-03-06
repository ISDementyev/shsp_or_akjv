import nltk
import numpy as np
from utils import read_corpus, process_string, count_freqs, train_naive_bayes, test_naive_bayes, naive_bayes_predict

# download necessary data
nltk.download("punkt_tab")
nltk.download("stopwords")

# initialize processed KJV
print("1. Processing AKJV")
akjv_file = "/Users/ilya/Desktop/DL_AI/shsp_or_kjb/kjb_corpus/AKJV.txt"
akjv = read_corpus(akjv_file)
processed_words_akjv = process_string(akjv, kjv=True)
print("2. Finished processing AKJV")
# print(processed_words_akjv[:20])

# initialize processed Shakespeare
print("3. Processing Shakespeare")
shsp_file = "/Users/ilya/Desktop/DL_AI/shsp_or_kjb/shsp/Paragraphs.txt"
shsp = read_corpus(shsp_file)
processed_words_shsp = process_string(shsp, kjv=False)
print(processed_words_shsp[:75])
print("4. Finished processing Shakespeare")

# From KJV: Label 1
# From Shakespeare: Label 0
print("5. Creating freqs dict")
freqs = dict()
freqs = count_freqs(freqs, processed_words_akjv, kjv=True)
freqs_all = count_freqs(freqs, processed_words_shsp, kjv=False)
print(freqs[("food", 1)])
print("6. Finished creating freqs dict")

N_sh = len(processed_words_shsp)
N_akjv = len(processed_words_akjv)

print(N_sh, N_akjv)

# divide shakespeare into training and testing
print("7. Dividing train and test splits")
train_shsp = processed_words_shsp[:int(0.8 * N_sh)] # 80% train
test_shsp = processed_words_shsp[int(0.8 * N_sh):] # 20% test

# divide AKJV into training and testing
train_akjv = processed_words_akjv[:int(0.8 * N_akjv)] # see above
test_akjv = processed_words_akjv[int(0.8 * N_akjv):] # see above

# make test and train sets for features
train_x = train_shsp + train_akjv
test_x = test_shsp + test_akjv

# make test and train sets for labels
train_y = np.append(np.zeros(len(train_shsp)), np.ones(len(train_akjv)))
test_y = np.append(np.zeros(len(test_shsp)), np.ones(len(test_akjv)))
print("8. Finished dividing train and test splits")

print("9. Training Naive Bayes")
logprior, loglikelihood = train_naive_bayes(freqs, train_x, train_y)
print("10. Finished training Naive Bayes")

print(f"11. Shakespeare prediction: {naive_bayes_predict("wherefore art thou Romeo?", logprior, loglikelihood) < 0}")
print(f"12. Bible Prediction: {naive_bayes_predict("It is easier for a camel to go through the eye of a needle than for a rich man to enter the kingdom of God", logprior, loglikelihood) > 0}")
# print(f"13. Naive Bayes Accuracy: {test_naive_bayes(test_x, test_y, logprior, loglikelihood)}")