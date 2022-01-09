import numpy as np
import re
from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer

porter=PorterStemmer()
sentiments = []
reviews = []
test_reviews = []



def stemSentence(sentence):
    token_words=word_tokenize(sentence)
    token_words
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)



# gets cosine similarity of the test data and their sentiment and k (odd int) and returns the result for the test 
def answer(cos_similarity, sentiments, k):
    answer = []
    pos_counter = 0
    neg_counter = 0
    result = 0
    for row in cos_similarity:
        pos_counter = 0
        neg_counter = 0        
        row = np.array(row)
        # to find k largest values (most similar) indexes so we could get their sentiments
        k_largest_cos_indices = np.argpartition(-row, range(0,k))[:k]
        for i in k_largest_cos_indices:
            if(sentiments[int(i)]) == '-1':
                neg_counter += 1
            else:
                pos_counter += 1
        if pos_counter > neg_counter:
            result = '+1'
        else:
            result = '-1'
        
        answer.append(result)
    return answer



# clean the training file
training_data_file_obj = open("train_data.txt","r")
training_data_list = training_data_file_obj.readlines()
for i in range(1, len(training_data_list)):
    rev = training_data_list[i]
   
    sentiments.append(rev[0:2])
    soup = BeautifulSoup(rev[2: len(rev)], 'lxml')
  
    alphabet = re.sub("[^a-zA-Z]", " ", soup.get_text())
    lower_alphabet = alphabet.lower()
        
    stemSent = stemSentence(lower_alphabet)
        
    words = stemSent.split()
    stop_words = set(stopwords.words("english"))
    pure_words = [word for word in words if not word in stop_words]
    
    spaced_worded_review = (" ".join(pure_words))
    reviews.append(spaced_worded_review)


#clean the test file
test_data_file_obj = open("test_data.txt","r")
test_data_list = test_data_file_obj.readlines()

for k in range(len(test_data_list)):
    test_rev = test_data_list[k]
    soup = BeautifulSoup(test_rev, 'lxml')
    alphabet = re.sub("[^a-zA-Z]", " ", soup.get_text())
    lower_alphabet = alphabet.lower()
    stemSent = stemSentence(lower_alphabet)
    words = stemSent.split()
    # remove stop words
    stop_words = set(stopwords.words("english"))
    pure_words = [word for word in words if not word in stop_words]
    # create a list of pure clean space seperated reviews
    spaced_worded_test_review = (" ".join(pure_words))
    test_reviews.append(spaced_worded_test_review)



vectorizer = TfidfVectorizer(sublinear_tf = True, ngram_range=(1,2), max_features = 10000)

# fit_transform is used to get normalized vector out of our reviews
training_matrix = vectorizer.fit_transform(reviews)
training_matirx = training_matrix.toarray()

# transform is used in order to get compatible dimentions.
testing_matrix = vectorizer.transform(test_reviews)
testing_matrix = testing_matrix.toarray()


# cosine similarity matrix
cos_similarity = cosine_similarity(  testing_matrix, training_matrix)


# k is square root of 15000 which is number of our training data points
output = answer(cos_similarity, sentiments, 123)
with open("format.txt",'w') as out:
    for line in output:
        out.write(line+ '\n')