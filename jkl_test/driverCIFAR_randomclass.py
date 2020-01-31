#File to run the machine learning expts on Lifelong Forests
import lifelong_forests
import numpy as np
import sklearn.datasets
import pickle

#Get all of the training data staged
train_file = 'C:\\LifelongLearning\\lifelong-learning\\cef\\cifar100\\cifar-100-python\\train'
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

unpickled_train = unpickle(train_file)
train_keys = list(unpickled_train.keys())

train_data = unpickled_train[b'data']
class_idx_hash = {}
for cntr in range(0, len(unpickled_train[b'fine_labels'])):
    if unpickled_train[b'fine_labels'][cntr] not in class_idx_hash:
        class_idx_hash[unpickled_train[b'fine_labels'][cntr]] = []
    class_idx_hash[unpickled_train[b'fine_labels'][cntr]].append(cntr)

#Sqrt of number of training examples?  
n_trees = int(np.sqrt(len(class_idx_hash[0])))

#Get all of the test data staged
test_file = 'C:\\LifelongLearning\\lifelong-learning\\cef\\cifar100\\cifar-100-python\\test'
unpickled_test = unpickle(test_file)
test_keys = list(unpickled_test.keys())

test_class_idx_hash = {}
for cntr in range(0, len(unpickled_test[b'fine_labels'])):
    if unpickled_test[b'fine_labels'][cntr] not in test_class_idx_hash:
        test_class_idx_hash[unpickled_test[b'fine_labels'][cntr]] = []
    test_class_idx_hash[unpickled_test[b'fine_labels'][cntr]].append(cntr)

#Create the LLF
lifelong_forest = lifelong_forests.LifelongForest()

#Choose ten classes to discriminate against... randomly permute which 10 we choose
CLASSNUM = 10
accuracyDF = []
accuracyLLF = []
accuracyOrig = []
for iter in range(0, 100, 1):
    print('Counter: ' + str(iter))
    #Randomly choose 10 classes
    idxs = np.random.choice(100, size=CLASSNUM, replace=False)

    #Get the indices that we are going to train on for this round
    lstindices = []
    for cntr in idxs:
        lstindices += class_idx_hash[cntr]

    #Slice out just the training data for these classes - 500 examples per class on train
    X = train_data[lstindices]

    #Get the right labels to match the training data as well
    labelsX = []
    for idx in range(0, len(lstindices)):
        labelsX.append(unpickled_train[b'fine_labels'][lstindices[idx]])
    
    #Next get all the test data - 100 examples per class
    lstindicestest = []
    for cntr in idxs:
        lstindicestest += test_class_idx_hash[cntr]
    Xtest = unpickled_test[b'data'][lstindicestest]
    #And get all the test labels staged
    labelsXtest = []
    for idx in range(0, len(lstindicestest)):
        labelsXtest.append(unpickled_test[b'fine_labels'][lstindicestest[idx]])

    if iter == 0:
        #Capture very first X - to calculated the reverse transfer to the very first set of 10
        veryfirstXTest = Xtest
        veryfirstXTestLabels = labelsXtest

    #For each iteration, create a new DF in the LLF for the current iteration (set of random 10 classes)
    lifelong_forest.new_forest(X, labelsX, n_estimators=n_trees)

    #Capture how many DFs we've made so far
    total_forests = lifelong_forest.n_tasks

    #First always calculate the DF on the most recent model.
    df_task1=lifelong_forest.predict(Xtest, representation=total_forests-1, decider=total_forests-1)
    dfcurrent = np.sum(df_task1 == labelsXtest)/len(labelsXtest)
    print('DF iter:' + str(iter) + ':' + str(dfcurrent))
    if iter == 0:
        #If on the very first iteration, make sure to cache the DF score on the first model so we can see how we improve when using LLF
        origDFScore = dfcurrent
    
    if total_forests > 1:
        #If there are more than 1 forest, then start calculating the LLF.  First, calculate the LLF on the most recent iteration (not for reverse transfer)
        llf_task1=lifelong_forest.predict(Xtest, representation='all', decider=total_forests-1)
        accuracyLLF.append(np.sum(llf_task1 == labelsXtest)/len(labelsXtest))
        accuracyDF.append(np.sum(df_task1 == labelsXtest)/len(labelsXtest))
        
        #Then calculate the LLF on the ORIGINAL example to calculate reverse transfer- THIS is the accuracy array that matters.
        llf_task_origx =lifelong_forest.predict(veryfirstXTest, representation='all', decider=0)
        accuracyOrig.append(np.sum(llf_task_origx == veryfirstXTestLabels)/len(veryfirstXTestLabels))
    print('DF:' + str(accuracyDF))
    print('LLF:' + str(accuracyLLF))
    print('Orig LLF:' + str(accuracyOrig))
    print('Orig DF:' + str(origDFScore))
    #For each iteration, rewrite out all of our scores.  We only really care about accuracyOrig, but keeping the others for convenience as well.
    fout = open("scores.csv", 'w')
    fout.write('DF,LLF,OrigLLF\n')
    for cntr in range(0, len(accuracyOrig)):
        fout.write(str(accuracyDF[cntr]) + ',' + str(accuracyLLF[cntr]) + ',' + str(accuracyOrig[cntr]) + '\n')
    fout.close()