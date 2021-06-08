


# # Predicting if a text is spam through data training 
# 
# In this assignment you will explore text message data and create models to predict if a message is spam or not. 

# In[1]:


import pandas as pd
import numpy as np

spam_data = pd.read_csv('spam.csv')

spam_data['target'] = np.where(spam_data['target']=='spam',1,0)
spam_data.head(10)


# In[2]:


from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(spam_data['text'], 
                                                    spam_data['target'], 
                                                    random_state=0)


# ### Question 1
# What percentage of the documents in `spam_data` are spam?
# 
# *This function should return a float, the percent value (i.e. $ratio * 100$).*

# In[26]:


def answer_one():
    
    
    return (len(spam_data[spam_data['target']==1])/len(spam_data))*100


# In[27]:


answer_one()


# ### Question 2
# 
# Fit the training data `X_train` using a Count Vectorizer with default parameters.
# 
# What is the longest token in the vocabulary?
# 
# *This function should return a string.*

# In[5]:


from sklearn.feature_extraction.text import CountVectorizer

def answer_two():
    vect=CountVectorizer().fit(X_train)
    #vocab=vect.get_feature_names()
    [(len(i),i) for i in vect.get_feature_names()]
    
    return sorted([(len(i),i) for i in vect.get_feature_names()],reverse=True)[0][1]


# In[6]:


answer_two()


# ### Question 3
# 
# Fit and transform the training data `X_train` using a Count Vectorizer with default parameters.
# 
# Next, fit a fit a multinomial Naive Bayes classifier model with smoothing `alpha=0.1`. Find the area under the curve (AUC) score using the transformed test data.
# 
# *This function should return the AUC score as a float.*

# In[7]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer
def answer_three():
    vect=CountVectorizer().fit(X_train)
    X_train_transformed=vect.transform(X_train)
    model=MultinomialNB(alpha=0.1).fit(X_train_transformed,y_train)
    predictions=model.predict(vect.transform(X_test))
    auc=roc_auc_score(y_test,predictions)
    return auc


# In[8]:


answer_three()


# ### Question 4
# 
# Fit and transform the training data `X_train` using a Tfidf Vectorizer with default parameters.
# 
# What 20 features have the smallest tf-idf and what 20 have the largest tf-idf?
# 
# Put these features in a two series where each series is sorted by tf-idf value and then alphabetically by feature name. The index of the series should be the feature name, and the data should be the tf-idf.
# 
# The series of 20 features with smallest tf-idfs should be sorted smallest tfidf first, the list of 20 features with largest tf-idfs should be sorted largest first. 
# 
# *This function should return a tuple of two series
# `(smallest tf-idfs series, largest tf-idfs series)`.*

# In[9]:


from sklearn.feature_extraction.text import TfidfVectorizer

def answer_four():
    tf_vect=TfidfVectorizer().fit(X_train)
    X_train_transformed=tf_vect.transform(X_train)
    tfid_sorted=X_train_transformed.max(0).toarray()[0].argsort()
    feature_names=np.array(tf_vect.get_feature_names())
    return  (pd.Series(feature_names[tfid_sorted[:20]]).sort_values(),pd.Series(feature_names[tfid_sorted[:-21:-1]]).sort_values()) #Your answer here


# In[10]:


answer_four()


# ### Question 5
# 
# Fit and transform the training data `X_train` using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than **3**.
# 
# Then fit a multinomial Naive Bayes classifier model with smoothing `alpha=0.1` and compute the area under the curve (AUC) score using the transformed test data.
# 
# *This function should return the AUC score as a float.*

# In[28]:


def answer_five():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.metrics import roc_auc_score
    tf_vect=TfidfVectorizer(min_df=3).fit(X_train)
    X_train_transformed=tf_vect.transform(X_train)
    nb_classifier=MultinomialNB(alpha=0.1)
    model=nb_classifier.fit(X_train_transformed,y_train)
    predictions=model.predict(tf_vect.transform(X_test))
    return roc_auc_score(y_test,predictions)


# In[29]:


answer_five()


# ### Question 6
# 
# What is the average length of documents (number of characters) for not spam and spam documents?
# 
# *This function should return a tuple (average length not spam, average length spam).*

# In[13]:


def answer_six():
    spam_data['ave_len']=pd.Series(len(spam_data['text'][i]) for i in range(0,len(spam_data)))
    np.where(spam_data['target']==0,spam_data['ave_len'],0).sum()
    return (np.where(spam_data['target']==0,spam_data['ave_len'],0).sum()/len(spam_data[spam_data['target']==0]),np.where(spam_data['target']==1,spam_data['ave_len'],0).sum()/len(spam_data[spam_data['target']==1]))#Your answer here


# In[14]:


answer_six()


# <br>
# <br>
# The following function has been provided to help you combine new features into the training data:

# In[15]:


def add_feature(X, feature_to_add):
    """
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    """
    from scipy.sparse import csr_matrix, hstack
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')


# ### Question 7
# 
# Fit and transform the training data X_train using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than **5**.
# 
# Using this document-term matrix and an additional feature, **the length of document (number of characters)**, fit a Support Vector Classification model with regularization `C=10000`. Then compute the area under the curve (AUC) score using the transformed test data.
# 
# *This function should return the AUC score as a float.*

# In[36]:


from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score 
def answer_seven():
    tfid_vect=TfidfVectorizer(min_df=5).fit(X_train)
    X_train_transformed=tfid_vect.transform(X_train)
    X_train_textlen_added=add_feature(X_train_transformed,pd.Series(len(pd.DataFrame(X_train).iloc[i]) for i in range(0,len(X_train))))
    svc_model=SVC(C=10000)
    model=svc_model.fit(X_train_textlen_added,y_train)
    predictions=model.predict(add_feature(tfid_vect.transform(X_test),pd.Series(len(pd.DataFrame(X_test).iloc[i]) for i in range(0,len(X_test)))))
    return roc_auc_score(y_test,predictions)


# In[37]:


answer_seven()


# ### Question 8
# 
# What is the average number of digits per document for not spam and spam documents?
# 
# *This function should return a tuple (average # digits not spam, average # digits spam).*

# In[38]:


def answer_eight():
    import re
    spam_data['digits_per_doc']=pd.Series([ len(re.findall(r'\d',spam_data.loc[i,'text'])) for i in range(0,len(spam_data))])
    return np.where(spam_data['target']==0,spam_data['digits_per_doc'],0).sum()/len(spam_data[spam_data['target']==0]),np.where(spam_data['target']==1,spam_data['digits_per_doc'],0).sum()/len(spam_data[spam_data['target']==1])#Your answer here


# In[39]:


answer_eight()


# ### Question 9
# 
# Fit and transform the training data `X_train` using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than **5** and using **word n-grams from n=1 to n=3** (unigrams, bigrams, and trigrams).
# 
# Using this document-term matrix and the following additional features:
# * the length of document (number of characters)
# * **number of digits per document**
# 
# fit a Logistic Regression model with regularization `C=100`. Then compute the area under the curve (AUC) score using the transformed test data.
# 
# *This function should return the AUC score as a float.*

# In[40]:


from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
import re
def answer_nine():
    vect=TfidfVectorizer(min_df=5,ngram_range=(1,3)).fit(X_train)
    X_train_transformed=vect.transform(X_train)
    X_train_len_dig_added=add_feature(add_feature(X_train_transformed,pd.Series(len(pd.DataFrame(X_train).iloc[i]) for i in range(0,len(X_train)))),pd.Series([len(re.findall(r'\d',pd.DataFrame(X_train).iloc[i][0])) for i in range(0,len(X_train))]))
    log_reg=LogisticRegression(C=100)
    model=log_reg.fit(X_train_len_dig_added,y_train)
    predictions=model.predict(add_feature(add_feature(vect.transform(X_test),pd.Series(len(pd.DataFrame(X_test).iloc[i]) for i in range(0,len(X_test)))),pd.Series([len(re.findall(r'\d',pd.DataFrame(X_test).iloc[i][0])) for i in range(0,len(X_test))])))
    
    return roc_auc_score(y_test,predictions)#Your answer here


# In[41]:


answer_nine()


# ### Question 10
# 
# What is the average number of non-word characters (anything other than a letter, digit or underscore) per document for not spam and spam documents?
# 
# *Hint: Use `\w` and `\W` character classes*
# 
# *This function should return a tuple (average # non-word characters not spam, average # non-word characters spam).*

# In[22]:


import re
def answer_ten():
    spam_data['len_of_nonword']=pd.Series([len(re.findall(r'\W',spam_data.loc[i,'text'])) for i in range(0,len(spam_data))])
    not_spam_ave=np.where(spam_data['target']==0,spam_data['len_of_nonword'],0).sum()/len(spam_data[spam_data['target']==0])
    spam_ave=np.where(spam_data['target']==1,spam_data['len_of_nonword'],0).sum()/len(spam_data[spam_data['target']==1])
    return (not_spam_ave,spam_ave)#Your answer here


# In[23]:


answer_ten()


# ### Question 11
# 
# Fit and transform the training data X_train using a Count Vectorizer ignoring terms that have a document frequency strictly lower than **5** and using **character n-grams from n=2 to n=5.**
# 
# To tell Count Vectorizer to use character n-grams pass in `analyzer='char_wb'` which creates character n-grams only from text inside word boundaries. This should make the model more robust to spelling mistakes.
# 
# Using this document-term matrix and the following additional features:
# * the length of document (number of characters)
# * number of digits per document
# * **number of non-word characters (anything other than a letter, digit or underscore.)**
# 
# fit a Logistic Regression model with regularization C=100. Then compute the area under the curve (AUC) score using the transformed test data.
# 
# Also **find the 10 smallest and 10 largest coefficients from the model** and return them along with the AUC score in a tuple.
# 
# The list of 10 smallest coefficients should be sorted smallest first, the list of 10 largest coefficients should be sorted largest first.
# 
# The three features that were added to the document term matrix should have the following names should they appear in the list of coefficients:
# ['length_of_doc', 'digit_count', 'non_word_char_count']
# 
# *This function should return a tuple `(AUC score as a float, smallest coefs list, largest coefs list)`.*

# In[24]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import re
def answer_eleven():
    vect=CountVectorizer(min_df=5,ngram_range=(2,5),analyzer='char_wb').fit(X_train)
    X_train_vectorized=vect.transform(X_train)
    length_of_doc=pd.Series(len(pd.DataFrame(X_train).iloc[i]) for i in range(0,len(X_train)))
    digit_count=pd.Series([len(re.findall(r'\d',pd.DataFrame(X_train).iloc[i][0])) for i in range(0,len(X_train))])
    non_word_char_count=pd.Series([len(re.findall(r'\W',pd.DataFrame(X_train).iloc[i][0])) for i in range(0,len(X_train))])
    X_train_all3=add_feature(add_feature(add_feature(X_train_vectorized,length_of_doc),digit_count),non_word_char_count)
    log_reg=LogisticRegression(C=100)
    model=log_reg.fit(X_train_all3,y_train)
    X_test_len_dig_added=add_feature(add_feature(vect.transform(X_test),pd.Series(len(pd.DataFrame(X_test).iloc[i]) for i in range(0,len(X_test)))),pd.Series([len(re.findall(r'\d',pd.DataFrame(X_test).iloc[i][0])) for i in range(0,len(X_test))]))
    X_test_all3_added=add_feature(X_test_len_dig_added,pd.Series([len(re.findall(r'\W',pd.DataFrame(X_test).iloc[i][0])) for i in range(0,len(X_test))]))
    predictions=model.predict(X_test_all3_added)
    auc_score=roc_auc_score(y_test,predictions)
    feature_names=np.array(vect.get_feature_names())
    feature_names=np.append(feature_names,['length_of_doc', 'digit_count', 'non_word_char_count'])
    sorted_coef=model.coef_[0].argsort()
    return  (auc_score,list(feature_names[sorted_coef[:10]]), list(feature_names[sorted_coef[:-11:-1]])) #feature_names[sorted_coef[:10]]#Your answer here


# In[25]:


answer_eleven()

