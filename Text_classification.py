#!/usr/bin/env python
# coding: utf-8

# ## Assignment 3c

# #### Submitted by:
# 
# #### **Calvin Smith**
# 
# #### **Bragadesh Bharatwaj Sundararaman**
# 
# #### **Amogha Udayakumar**

# ### Loading and cleaning data

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#Splitting into training and test set

train = pd.read_csv('PA3_train.tsv',sep = '\t',header = None)
test = pd.read_csv('PA3_test_clean.tsv',sep = '\t',header = None)


# In[3]:


total_len = len(train)

print(len(train))
print(len(test))


# In[4]:


# Changing columns names to something better: 

train.rename(columns = {0:'annot',1:'text'},inplace = True)
test.rename(columns = {0:'label',1:'text'},inplace = True)


# In[5]:


# Removing the instances where the annotators do not agree
train = train.loc[(train['annot'] == '0/0') | (train['annot'] == '1/1')]


# In[6]:


# Number of observations removed 

print(total_len - len(train))


# In[7]:


# Creating new labels as 0 or 1 instead of 0/0 or 1/1

train['label'] = len(train)*0
for i in range(len(train)):
    if train['annot'].iloc[i] == '1/1':
        train['label'].iloc[i] = 1


# In[8]:


# class balance training set

print(len(train[train['label']== 0])/len(train))
print(len(train[train['label']== 1])/len(train))


# In[9]:


# class balance testing set

print(len(test[test['label']== 0])/len(test))
print(len(test[test['label']== 1])/len(test))


# In[10]:


# dividing the training data into new training and validation sets for model building

train_temp, validation = train_test_split(train,test_size = 0.2)


# In[11]:


len(train[train['label'] == 0])/len(train)


# In[12]:


len(train_temp[train_temp['label'] == 0])/len(train_temp)


# In[13]:


len(validation[validation['label'] == 0])/len(validation)


# # Naive Bayes

# In[14]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# In[15]:


# Function for plotting the confusion matrix using Seaborn

def confusion_matrix_plot(cf_matrix):
    ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues', fmt='d')
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')
    plt.show()


# In[16]:


# Function where we can try different Naive Bayes model and evaluate the accuracy before moving
# on to the test set.

def naive_bayes(train,validation,model_type,n_range,Alpha = 1.0,stopwords = []):
    
    tfidf = TfidfVectorizer(stop_words = stopwords,ngram_range = n_range)
    
    Xtrain = tfidf.fit_transform(train['text'])
    Xtest = tfidf.transform(validation['text'])

    if model_type == "multinomial":

        multi_NB = MultinomialNB(alpha = Alpha)
        clf = multi_NB.fit(Xtrain,train['label'])
        pred = clf.predict(Xtest)

        acc = accuracy_score(validation['label'],pred)
        conf_mat = confusion_matrix(validation['label'],pred)
        acc_class = [conf_mat[0,0]/(conf_mat[0,0]+conf_mat[1,0]),conf_mat[1,1]/(conf_mat[1,1]+conf_mat[0,1])]
     
    
    if model_type == "bernoulli":

        ber_NB = BernoulliNB(alpha = Alpha)
        clf = ber_NB.fit(Xtrain,train['label'])
        pred = clf.predict(Xtest)

        acc = accuracy_score(validation['label'],pred)
        conf_mat = confusion_matrix(validation['label'],pred)
        acc_class = [conf_mat[0,0]/(conf_mat[0,0]+conf_mat[1,0]),conf_mat[1,1]/(conf_mat[1,1]+conf_mat[0,1])]

    return acc, pred, acc_class, conf_mat


# In[17]:


# Tuning the n-grams paramter for the TfidfVectorizer

acc_evaluation = []
best = -1
count = 1
for i in range(1,5):
    for j in range(count,5):
        acc, _, _, _= naive_bayes(train_temp,validation,'multinomial',(i,j))

        if acc > best:
            best = acc
            print(f'Current best: {best}')
            print(f'n_gram range = ({i},{j})')
        
    count = count + 1


# In[19]:


# Tuning the alpha parameter for naive Bayes classifier

alpha = [0.05,0.1,0.2,0.3,0.5,0.8,1.0,1.2,1.5,2,3,10,15,20]
best = -1
acc_list = []
for i in alpha:
    acc, _, _, _ = naive_bayes(train_temp,validation,'multinomial',(1,2),i,[])
    acc_list.append(acc)


# In[20]:


#Plotting Accuracy vs Alpha

plt.plot(alpha,acc_list)
plt.title(r'Accuracy as a function of $\alpha$')
plt.xlabel(r'$\alpha$')
plt.ylabel('Accuracy')
plt.show()


# In[21]:


# Function that counts the frequency of ngrams from a text corpus.
# X = text corpus
# n = number of common and uncommon words to output
# ngram = should output be unigram, bigram, trigram etc.

def get_words(X,n,ngram):
  vect = CountVectorizer(ngram_range = ngram)  
  X_counts = vect.fit_transform(X.text)

  sum_words = X_counts.sum(axis = 0)

  words_freq = [(word, sum_words[0, idx]) for word, idx in     vect.vocabulary_.items()]
  words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

  return words_freq[:n], words_freq[-n:]


# In[22]:


# Top 10 most common and uncommon words in the training data

top, bottom = get_words(train,10,(1,1))
print('Top 10 most common words:\n', top)
print('\n Top 10 most uncommon words:\n', bottom)


# In[24]:


# Final evaluation

acc, pred,acc_class, cf_matrix = naive_bayes(train,test,'multinomial',(1,2),Alpha=0.2)
print(f' Accuracy: {acc} \n Class accuracy: {acc_class}')
confusion_matrix_plot(cf_matrix)


# In[25]:


# To find instances where our model makes mistakes

index_list = []
for i in range(len(pred)):
    if pred[i] != test['label'].iloc[i]:
        index_list.append(i)


# In[26]:


# Examples of the text where the model made mistakes

print('Actual Value:',test['label'].iloc[index_list[5]],'Prediction:',pred[index_list[5]],'\nText:',test['text'].iloc[index_list[5]], '\n\nActual Value:',test['label'].iloc[index_list[5]],'Prediction:',pred[index_list[5]],'\nText:', test['text'].iloc[index_list[15]], '\n\nActual Value:',test['label'].iloc[index_list[5]],'Prediction:',pred[index_list[5]],'\nText:', test['text'].iloc[index_list[25]])


# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=190c7b43-b754-4fd5-a62f-1a140edc140a' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
