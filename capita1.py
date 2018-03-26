from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC, NuSVC
from sklearn import metrics
from sklearn.naive_bayes import BernoulliNB, GaussianNB
import unicodedata
import xlrd
import csv
import pandas as pd
import numpy as np
import sys, random
from nltk.stem.lancaster import LancasterStemmer
import nltk

def custom_vectorizer(answer):
    tbl = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))
    def remove_punctuation(text):
        return text.translate(tbl)
    stemmer = LancasterStemmer()
    data = answer
    words = []
    # a list of tuples with words in the sentence and category name 
    docs = []
    answer = remove_punctuation(answer)
    w = nltk.word_tokenize(answer)
    words.extend(w)
    docs.append(w)
    words = [stemmer.stem(w.lower()) for w in words]
    words = sorted(list(set(words)))

    return(words)
    ## create our training data
    #training = []
    #output = []
    ## create an empty array for our output
    #output_empty = [0] * len(categories)
    #for doc in docs:
    #    # initialize our bag of words(bow) for each document in the list
    #    bow = []
    #    # list of tokenized words for the pattern
    #    token_words = doc
    #    # stem each word
    #    token_words = [stemmer.stem(word.lower()) for word in token_words]
    #    # create our bag of words array
    #    for w in words:
    #        bow.append(1) if w in token_words else bow.append(0)

    #    output_row = list(output_empty)
    #    output_row[categories.index(doc[1])] = 1

    #    # our training set will contain a the bag of words model and the output row that tells which catefory that bow belongs to.
    #    training.append([bow, output_row])

class CustomVectorizer(CountVectorizer):
    def build_tokenizer(self):
        #tokenize = super(CustomVectorizer, self).build_tokenizer()
        return lambda doc: list(custom_vectorizer(doc))
        
def read_all_data(xlsxfilename, csv1, csv2, csv3):
    workbook = xlrd.open_workbook(xlsxfilename)
    worksheet = workbook.sheet_by_index(3)

    offset = 1
    rows = []
    for i, row in enumerate(range(worksheet.nrows)):
        if i <= offset:  # (Optionally) skip headers
            continue
        r = []
        for j, col in enumerate(range(worksheet.ncols)):
            r.append(worksheet.cell_value(i, j))
        rows.append(r)

    grades = {}
    for row in rows:
        _id = row[0].split()
        grades[_id[1]] = row[1:]

    df = pd.read_csv("fellows.csv", encoding='Latin-1')
    question1 = pd.read_csv(csv1, encoding='Latin-1')
    question2 = pd.read_csv(csv2, encoding='Latin-1')
    question9 = pd.read_csv(csv3, encoding='Latin-1')
    answers = list(df["Applicant_Name"]), list(df["question 1"]), list(question1["TOTALSCORE"]), list(df["question 2"]), list(question2["TOTALSCORE"]), list(df["question 9"]), list(question9["TOTALSCORE"])
    for i in range(len(answers[0])):
        _id = answers[0][i].split()
        _id = _id[1]
        grades[_id] = [(grades[_id][0], float(answers[2][i][:-1]), answers[1][i]), (grades[_id][1], float(answers[4][i][:-1]), answers[3][i]), (grades[_id][2], float(answers[6][i][:-1]), answers[5][i]), (grades[_id][3], 'total')]
    return grades

def capita_algorithm(training_const, clf, inf):
    grades = read_all_data('data.xlsx', 'resultsquestion 1.csv', 'resultsquestion 2.csv', 'resultsquestion 9.csv')

    #text_clf = Pipeline([('vect', CountVectorizer(analyzer='char_wb', ngram_range=(5,5))),
    #text_clf = Pipeline([('vect', CustomVectorizer()),
                        #('tfidf', TfidfTransformer()),
                        #('clf', SVC(kernel="poly", degree=3, probability=True))
                        #('clf', NuSVC(nu=0.07, kernel="poly", degree=3, probability=True))
                        #('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None))
                        #('clf', BernoulliNB())
                        #])
    text_clf = clf
    info = inf + [training_const]
    sentences = []
    classes = []

    #1 for curators 0 for word dataset
    question = 1
    inputs = list(grades.keys())
    random.shuffle(inputs)
    for applicant in inputs:
        classes.append(grades[applicant][question][1])
        sentences.append(grades[applicant][question][2])

    categories = set(classes)
    #print(sentences)
    #print(classes)
    training = sentences[:training_const]
    training_classes = classes[:training_const]
    test = sentences[training_const:]
    test_classes = classes[training_const:]
    #classes = []
    #for item in range(len(categories)):
        #for _item in data[categories[item]]:
            #sentences.append(_item)
            #classes.append(item)


    text_clf.fit(training, training_classes)  
    predicted = text_clf.predict(test)
    #print(list(map(lambda x: categories[x], predicted)))
    #print(predicted)
    #print(len(categories))
    #print(list(map(lambda x,y: (x-y)/len(categories), predicted, test_classes)))
    error = list(map(lambda x,y: abs((x-y))/(max(categories) - min(categories)), predicted, test_classes))
    #print(error)
    avg_error = sum(error)/len(error)
    #print(metrics.classification_report(test_classes, predicted))
    results = metrics.classification_report(test_classes, predicted)
    results = results.split()
    with open('algorithmresultsquestion{0}new.log'.format(question), 'a') as f:
        f.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5:.3f}\t{6:.3f}\n".format(info[0], info[1], info[2], results[-4], results[-3], np.mean(predicted == test_classes), avg_error))
    return results

text_clf_list = [#Pipeline([('vect', CustomVectorizer()),
                    #('tfidf', TfidfTransformer()),
                    #('clf', SVC(kernel="poly", degree=3, probability=True))
                    #]),
                #Pipeline([('vect', CustomVectorizer()),
                    #('tfidf', TfidfTransformer()),
                    #('clf', NuSVC(nu=0.07, kernel="poly", degree=3, probability=True))
                    #]),
                #Pipeline([('vect', CustomVectorizer()),
                    #('tfidf', TfidfTransformer()),
                    #('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None))
                    #]),
                Pipeline([('vect', CustomVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf', BernoulliNB())
                    ])]

info = [#['SVC kernel=poly, degree=3, probability=True', 'CustomVectorizer'],
        #['NuSVC nu=0.07, kernel=poly, degree=3, probability=True', 'CustomVectorizer'],
        #['SGDClassifier loss=hinge, penalty=l2, alpha=1e-3, random_state=42, max_iter=5, tol=None', 'CustomVectorizer'],
        #['GaussianNB', 'CustomVectorizer'],
        ['BernoulliNB', 'CustomVectorizer']]

for clf, inf in zip(text_clf_list, info):
    for tset in [10, 30, 60, 80, 100, 150]:
        print('training size {0}'.format(tset))
        l = list(range(10))
        for i in l:
            print('repetition {0}'.format(i), end='\r', flush = True)
            try:
                res = capita_algorithm(tset, clf, inf)
            except ValueError:
                l.append(i)
