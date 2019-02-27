from gensim.models import Word2Vec
import numpy as np
import sklearn as sk
import random

#concepts
CAPITALISM = "capitalism"
#NFolds
KFOLDS = 5

#load model
model = Word2Vec.load("model1")

CONCEPTS = {
"capitalism": ["trade","monetary","money","financial","finance","labor","class","working","class","proletariat","bourgeois","capital",
    "production","industry","industrialization","commerce","commercial","factory","factories","laborers","laborer","workers","urbanization","urban","capitalism","consumption","consume","goods"]    


}


positiveN = len(CONCEPTS["capitalism"])


#make X with positive examples
X = np.array([])
for word in CONCEPTS[CAPITALISM]:
    if word in model:
        if X.shape == (0,):
            X = model.wv[word][np.newaxis,:]
        else:
            print(X.shape, model.wv[word].shape)
            X = np.concatenate((X, model.wv[word][np.newaxis,:]), axis=0)
        
    else:
        continue
positiveN = X.shape[0]
#print(X)

#add to X negative examples
for i in range(positiveN):
    randomWord = random.choice(list(model.wv.vocab.items()))[0]
    X = np.concatenate((X, model.wv[randomWord][np.newaxis,:]), axis=0)

print("size of full matrix: ", X.shape)

#make y values
y = [1]*positiveN + [0]*positiveN
y = np.array(y)

#kfold indices
from sklearn.model_selection import KFold
kf = KFold(n_splits=KFOLDS)
foldsplits = list(kf.split(X))

#run SVM for every train-val pair

from sklearn.svm import SVC
#svm = SVC(gamma="auto")
for i,pair in enumerate(foldsplits):
    print(pair)
    trainX = np.take(X, pair[0], axis=0)
    trainy = np.take(y, pair[0], axis=0)
    valX = np.take(X, pair[1], axis=0)
    valy = np.take(y, pair[1], axis=0)
    svm = SVC(gamma="auto")
    svm.fit(trainX,trainy)
    print(valX)
    yPred = svm.predict(valX)
    print(valy)
    acc = np.sum(yPred== valy)/len(valy)
    print(str(i) + ": ", acc)
#
#svm = SVC(gamma="auto")
#svm.fit(X,y)
#





