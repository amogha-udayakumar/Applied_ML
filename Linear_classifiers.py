#!/usr/bin/env python
# coding: utf-8

# ## DAT340/DIT867 Programming assignment 4: Implementing linear classifiers
# 
# #### Calvin Smith
# #### Bragadesh Bharatwaj Sundararaman
# #### Amogha Udayakumar
# 
# 

# ### Exercise question:
# 
# In the pipeline, DictVectorizer is used. It converts the string input into vectors which are then input to the perceptron. When we look closely at the output of the dictVectorizer we can find that in the first case the input is clearly linearly separable. Thus the perceptron is able to perfectly classify it. On the other hand, in the second training data, when we analyze the dictvectorizer output we can see that the data is not linearly separable. This is the reason why even if we utilize Linear SVC it still fails. In order to classify linearly inseparable data, we can either use a non-linear classifier or if using a linear classifier we have to increase the dimensions of the input i.e. add more features to the input by creating new features using existing ones.
# 

# ### Pegasos classes:

# In[3]:


import numpy as np
from numpy import linalg as LA
from scipy.linalg.blas import ddot
from scipy.linalg.blas import dscal
from scipy.linalg.blas import daxpy
from sklearn.base import BaseEstimator

class LinearClassifier(BaseEstimator):
    """
    General class for binary linear classifiers. Implements the predict
    function, which is the same for all binary linear classifiers. There are
    also two utility functions.
    """

    def decision_function(self, X):
        """
        Computes the decision function for the inputs X. The inputs are assumed to be
        stored in a matrix, where each row contains the features for one
        instance.
        """
        return X.dot(self.w)

    def predict(self, X):
        """
        Predicts the outputs for the inputs X. The inputs are assumed to be
        stored in a matrix, where each row contains the features for one
        instance.
        """

        # First compute the output scores
        scores = self.decision_function(X)

        # Select the positive or negative class label, depending on whether
        # the score was positive or negative.
        out = np.select([scores >= 0.0, scores < 0.0],
                        [self.positive_class,
                         self.negative_class])
        return out

    def find_classes(self, Y):
        """
        Finds the set of output classes in the output part Y of the training set.
        If there are exactly two classes, one of them is associated to positive
        classifier scores, the other one to negative scores. If the number of
        classes is not 2, an error is raised.
        """
        classes = sorted(set(Y))
        if len(classes) != 2:
            raise Exception("this does not seem to be a 2-class problem")
        self.positive_class = classes[1]
        self.negative_class = classes[0]

    def encode_outputs(self, Y):
        """
        A helper function that converts all outputs to +1 or -1.
        """
        return np.array([1 if y == self.positive_class else -1 for y in Y])


class Pegasos(LinearClassifier):
    """
    Implementation of the pegasos learning algorithm using hinge-loss function.
    """

    def fit(self, X, Y):
        """
        Train a linear classifier using the pegasos learning algorithm with hinge-loss.
        """

        # First determine which output class will be associated with positive
        # and negative scores, respectively.
        self.find_classes(Y)

        # Convert all outputs to +1 (for the positive class) or -1 (negative).
        Ye = self.encode_outputs(Y)

        # If necessary, convert the sparse matrix returned by a vectorizer
        # into a normal NumPy matrix.
        if not isinstance(X, np.ndarray):
            X = X.toarray()

        # Initialize the weight vector to all zeros.
        n_features = X.shape[1]
        self.w = np.zeros(n_features)

        
        # Pegasos algorithm
        
        # Number of pairs to be randomly selected
        T = 100000
        
        # Lambda
        Lambda = 1/T
        
        for t in range(1,T+1):
            i = np.random.randint(1,len(X))
            
            #learning rate
            nu = 1/(Lambda*t)

            x = X[i]
            y = Ye[i]
            score = np.dot(self.w,x)
            
            if y*score < 1:
                self.w = (1-nu*Lambda)*self.w + (nu*y)*x
            else:
                self.w = (1-nu*Lambda)*self.w
                
                
class Pegasos_BLAS(LinearClassifier):
    """
    A straightforward implementation of the pegasus learning algorithm using BLAS-functions
    """


    def fit(self, X, Y):
        """
        Train a linear classifier using the pegasos learning algorithm using BLAS-functions
        """

        # First determine which output class will be associated with positive
        # and negative scores, respectively.
        self.find_classes(Y)

        # Convert all outputs to +1 (for the positive class) or -1 (negative).
        Ye = self.encode_outputs(Y)

        # If necessary, convert the sparse matrix returned by a vectorizer
        # into a normal NumPy matrix.
        if not isinstance(X, np.ndarray):
            X = X.toarray()

        # Initialize the weight vector to all zeros.
        n_features = X.shape[1]
        self.w = np.zeros(n_features)

        
        # Number of pairs to be randomly selected
        T = 100000
        
        # Lambda
        Lambda = 1/T
                
        ### Pegasos algorithm using BLAS functions
        
        for t in range(1,T+1):
            i = np.random.randint(1,len(X))
            
            #learning rate
            nu = 1/(Lambda*t)

            x = X[i]
            y = Ye[i]
            score = ddot(self.w,x)
            
            if y*score < 1:
                #dscal(1-nu*Lambda,self.w)
                #daxpy(x,self.w,a = ddot(nu,y))
                daxpy(x,dscal(1-ddot(nu,Lambda),self.w),a=ddot(nu,y))
                                           
            else:           
                dscal(1-nu*Lambda,self.w)
                        
        
     
                       
class Pegasos_LR(LinearClassifier):
    """
    A straightforward implementation of the pegasos learning algorithm using log-loss.
    """

    def fit(self, X, Y):
        """
        Train a linear classifier using the pegasos learning algorithm with log-loss.
        """

        # First determine which output class will be associated with positive
        # and negative scores, respectively.
        self.find_classes(Y)

        # Convert all outputs to +1 (for the positive class) or -1 (negative).
        Ye = self.encode_outputs(Y)

        # If necessary, convert the sparse matrix returned by a vectorizer
        # into a normal NumPy matrix.
        if not isinstance(X, np.ndarray):
            X = X.toarray()

        # Initialize the weight vector to all zeros.
        n_features = X.shape[1]
        self.w = np.zeros(n_features)

        
        # Pegasos algorithm
        
        # Number of pairs to be randomly selected
        T = 100000
        epochs = np.round(T/10)
        
        # Lambda
        Lambda = 1/T
        
        sum_loss = 0
        
        for t in range(1,T+1):
            i = np.random.randint(1,len(X))
            #learning rate
            nu = 1/(Lambda*t)

            x = X[i]
            y = Ye[i]
            score = np.dot(self.w,x)
            
            loss = -(y*x)/(1+ np.exp(y*score))
            
            sum_loss = sum_loss + np.sum(loss)
            
            self.w = self.w - nu*(Lambda*self.w +loss)
            
            # printing current value of the objective function at each iteration.
            if t == epochs:
                epochs = epochs + 10000
                
                print(f'Objective function at {t}:{sum_loss/t + (Lambda/2)*((LA.norm(self.w))**2)}')
                   
                           
            
        

##### The following part is for the optional task.

### Sparse and dense vectors don't collaborate very well in NumPy/SciPy.
### Here are two utility functions that help us carry out some vector
### operations that we'll need.

def add_sparse_to_dense(x, w, factor):
    """
    Adds a sparse vector x, scaled by some factor, to a dense vector.
    This can be seen as the equivalent of w += factor * x when x is a dense
    vector.
    """
    w[x.indices] += factor * x.data

def sparse_dense_dot(x, w):
    """
    Computes the dot product between a sparse vector x and a dense vector w.
    """
    return np.dot(w[x.indices], x.data)




class SparsePegasos(LinearClassifier):
    """
    A straightforward implementation of the pegasos learning algorithm,
    assuming that the input feature matrix X is sparse.
    """


    def fit(self, X, Y):
        """
        Train a linear classifier using the pegasos learning algorithm.

        Note that this will only work if X is a sparse matrix, such as the
        output of a scikit-learn vectorizer.
        """
        self.find_classes(Y)

        # First determine which output class will be associated with positive
        # and negative scores, respectively.
        Ye = self.encode_outputs(Y)

        # Initialize the weight vector to all zeros.
        self.w = np.zeros(X.shape[1])

        # Iteration through sparse matrices can be a bit slow, so we first
        # prepare this list to speed up iteration.
        XY = list(zip(X, Ye))
        
        # number of iterations
        T = 100000
        
        # Lambda
        Lambda = 1/T
        
        for t in range(1,T+1):
            i = np.random.randint(1,len(XY))
            
            #learning rate
            nu = 1/(Lambda*t)

            x = XY[i][0]
            y = XY[i][1]
           
            score = sparse_dense_dot(x,self.w)
            
            if y*score < 1:
                self.w = (1-nu*Lambda)*self.w
                add_sparse_to_dense(x,self.w,nu*y)
            else:
                self.w = (1-nu*Lambda)*self.w
                

class SparsePegasos_scale(LinearClassifier):
    """
    A straightforward implementation of the pegasos learning algorithm,
    assuming that the input feature matrix X is sparse. 
    
    In this implementation we use a scaling trick to speed up the process.
    
    """

    def __init__(self, n_iter=20):
        """
        The constructor can optionally take a parameter n_iter specifying how
        many times we want to iterate through the training set.
        """
        self.n_iter = n_iter

    def fit(self, X, Y):
        """
        Train a linear classifier using the perceptron learning algorithm.

        Note that this will only work if X is a sparse matrix, such as the
        output of a scikit-learn vectorizer.
        """
        self.find_classes(Y)

        # First determine which output class will be associated with positive
        # and negative scores, respectively.
        Ye = self.encode_outputs(Y)

        # Initialize the weight vector to all zeros.
        self.w = np.zeros(X.shape[1])

        # Iteration through sparse matrices can be a bit slow, so we first
        # prepare this list to speed up iteration.
        XY = list(zip(X, Ye))
        
        #print(len(XY)) 
        T = 100000
        
        # Lambda
        Lambda = 1/T
        
        a = 1 
        
        for t in range(1,T+1):
            i = np.random.randint(1,len(XY))
            
            #learning rate
            nu = 1/(Lambda*t)

            x = XY[i][0]
            y = XY[i][1]
            
            a = (1-nu*Lambda)*a
           
            score = sparse_dense_dot(x,self.w)*a
            
            if y*score < 1:
                
                add_sparse_to_dense(x,self.w,(nu*y)/a)       
                
            else:
                
                self.w = self.w
          
        self.w = a*self.w


# ### Code to run the classifiers

# In[4]:


# In order to run the code the data needs to be in the same directory as the file. 


import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# This function reads the corpus, returns a list of documents, and a list
# of their corresponding polarity labels. 
def read_data(corpus_file):
    X = []
    Y = []
    with open(corpus_file, encoding='utf-8') as f:
        for line in f:
            _, y, _, x = line.split(maxsplit=3)
            X.append(x.strip())
            Y.append(y)
    return X, Y

def run_pegasos():
    
    # Read all the documents.
    X, Y = read_data('data/all_sentiment_shuffled.txt')
    
    # Split into training and test parts.
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2,
                                                    random_state=0)
    
    # Set up the preprocessing steps and the classifier.
    pipeline = make_pipeline(
        TfidfVectorizer(),
        SelectKBest(k=1000),
        Normalizer(),

        # Choose wich classifier to use
        Pegasos()
    )

    # Train the classifier.
    t0 = time.time()
    pipeline.fit(Xtrain, Ytrain)
    t1 = time.time()
    print('Training time: {:.2f} sec.'.format(t1-t0))

    # Evaluate on the test set.
    Yguess = pipeline.predict(Xtest)
    print('Accuracy: {:.4f}.'.format(accuracy_score(Ytest, Yguess)))
    
def run_pegasos_lr():
    
    # Read all the documents.
    X, Y = read_data('data/all_sentiment_shuffled.txt')
    
    # Split into training and test parts.
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2,
                                                    random_state=0)
    
    # Set up the preprocessing steps and the classifier.
    pipeline = make_pipeline(
        TfidfVectorizer(),
        SelectKBest(k=1000),
        Normalizer(),

        # Choose wich classifier to use
        Pegasos_LR()
    )

    # Train the classifier.
    t0 = time.time()
    pipeline.fit(Xtrain, Ytrain)
    t1 = time.time()
    print('Training time: {:.2f} sec.'.format(t1-t0))

    # Evaluate on the test set.
    Yguess = pipeline.predict(Xtest)
    print('Accuracy: {:.4f}.'.format(accuracy_score(Ytest, Yguess)))

def run_pegasos_BLAS():
    
    # Read all the documents.
    X, Y = read_data('data/all_sentiment_shuffled.txt')
    
    # Split into training and test parts.
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2,
                                                    random_state=0)
    
    # Set up the preprocessing steps and the classifier.
    pipeline = make_pipeline(
        TfidfVectorizer(),
        SelectKBest(k=1000),
        Normalizer(),

        # Choose wich classifier to use
        Pegasos_BLAS()
    )

    # Train the classifier.
    t0 = time.time()
    pipeline.fit(Xtrain, Ytrain)
    t1 = time.time()
    print('Training time: {:.2f} sec.'.format(t1-t0))

    # Evaluate on the test set.
    Yguess = pipeline.predict(Xtest)
    print('Accuracy: {:.4f}.'.format(accuracy_score(Ytest, Yguess)))


    
def run_pegasos_nosparse():
    
    # Read all the documents.
    X, Y = read_data('data/all_sentiment_shuffled.txt')
    
    # Split into training and test parts.
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2,
                                                    random_state=0)
    
    # Set up the preprocessing steps and the classifier.
    pipeline = make_pipeline(
        TfidfVectorizer(ngram_range = (1,2)),
        #SelectKBest(k=1000),
        Normalizer(),

        # Choose wich classifier to use
        Pegasos()
    )

    # Train the classifier.
    t0 = time.time()
    pipeline.fit(Xtrain, Ytrain)
    t1 = time.time()
    print('Training time: {:.2f} sec.'.format(t1-t0))

    # Evaluate on the test set.
    Yguess = pipeline.predict(Xtest)
    print('Accuracy: {:.4f}.'.format(accuracy_score(Ytest, Yguess)))
 

def run_sparse_pegasos():
    
    
    # Read all the documents.
    X, Y = read_data('data/all_sentiment_shuffled.txt')
    
    # Split into training and test parts.
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2,
                                                    random_state=0)
    
    # Set up the preprocessing steps and the classifier.
    pipeline = make_pipeline(
        TfidfVectorizer(ngram_range = (1,2)),
        #SelectKBest(k=1000),
        Normalizer(),

        # Choose wich classifier to use
        SparsePegasos()
    )

    # Train the classifier.
    t0 = time.time()
    pipeline.fit(Xtrain, Ytrain)
    t1 = time.time()
    print('Training time: {:.2f} sec.'.format(t1-t0))

    # Evaluate on the test set.
    Yguess = pipeline.predict(Xtest)
    print('Accuracy: {:.4f}.'.format(accuracy_score(Ytest, Yguess)))


        
    
def run_scale_pegasos():
    
    # Read all the documents.
    X, Y = read_data('data/all_sentiment_shuffled.txt')
    
    # Split into training and test parts.
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2,
                                                    random_state=0)
    
    # Set up the preprocessing steps and the classifier.
    pipeline = make_pipeline(
        TfidfVectorizer(ngram_range = (1,2)),
        #SelectKBest(k=1000),
        Normalizer(),

        # Choose wich classifier to use
        SparsePegasos_scale()
    )

    # Train the classifier.
    t0 = time.time()
    pipeline.fit(Xtrain, Ytrain)
    t1 = time.time()
    print('Training time: {:.2f} sec.'.format(t1-t0))

    # Evaluate on the test set.
    Yguess = pipeline.predict(Xtest)
    print('Accuracy: {:.4f}.'.format(accuracy_score(Ytest, Yguess)))



# ## MAIN:

# ### Question 1: Implementing the SVC Pegasos algorithm
# 
# We have chosen to follow the procedure in the paper by picking a fixed number T of randomly selected pairs to iterate through. We tried a few different values, and eventually chose T = 100 000 since it produced a fairly high accuracy and at a pretty good speed. Also, choosing a number higher than 100 000 did not lead to an increase in accuracy, but only an increase in time.
# 
# From this, Lambda was set to 1/T.
# 
# The code for the SVC pegasos classifier can be found in pegasos.py class Pegasos

# In[5]:


## Implementing the SVC pegasos algorithm

run_pegasos()


# ### Question 2: Implementing the LR Pegasos algorithm
# 
# We use the same number of iterations T and Lmabda as in the SVC case. 
# 
# The code for the LR pegasos classifier can be found in pegasos.py class Pegasos_LR.
# When running the code the Pegasos_LR class will also output the value of the objective function for every 10 000 iterations.

# In[6]:


## Implementing the LR pegasos algorithm

run_pegasos_lr()


# Using the log-loss function raises the accuarcy slightly (0.8254) but with an increased training time (5.73 seconds). This might be due to the fact that the log-loss uses more computations than the hinge-loss.

# ### Question 3
# 
# ### Bonus task 1: Making your code more efficient.

# #### a) Faster linear algebra operations
# 
# The code for the SVC pegasos classifier using BLAS functions can be found in pegasos.py class Pegasos_BLAS.

# In[7]:


## Implementing SVC pegasos algorithm using BLAS functions

run_pegasos_BLAS()


# Using the BLAS function helped speed up the linear algebra operations.
# In question 1, we got a training time of 3.22 seconds and an accuracy of 0.8212. Using BLAS functions we got a training time of 2.67 seconds and a similar accuracy of 0.8196.
# 

# #### b) Using sparse vectors
# 
# We start by running the original SVC pegasos from question 1 but this time without using SelectKbest and changing the TFIDF vectorizer ngram range to (1,2):

# In[4]:


## Implementing the SVC pegasos algorithm without Kbest and ngram_range = (1,2)
 
run_pegasos_nosparse()


# The accuracy has increased a bit, which is expected since we are utlizing a larger set of features. However, the training time has increased significantly!
# 
# Next step is to try the sparse version of SVC pegasos:
# 
# The code for the sparse SVC pegasos classifier can be found in pegasos.py class SparsePegasos.

# In[8]:


## Implementing the SVC pegasos algorithm using sparse vectors
## Remove SelectKBest 
## In the TFIDF-vectorizer, change ngram range to (1,2)

run_sparse_pegasos()


# By using sparse vectors we managed to decrease the training time from 465.57 seconds to 143.03 seconds while maintaining the accuracy.

# #### c) Speeding up the scaling operation
# 
# The code for the sparse SVC pegasos classifier with the scaling trick can be found in pegasos.py class SparsePegasos_scale.

# In[9]:


## Implementing the SVC pegasos algorithm using sparse vectors and scaling
## Remove SelectKBest 
## In the TFIDF-vectorizer, change ngram range to (1,2)

run_scale_pegasos()


# With the sclaing trick the training time was dramatically reduced, from 143.03 seconds to 7.78 seconds, again the accuracy is maintained at approximately the same level (it varies because of the randomness in sampling at each iteration T).
