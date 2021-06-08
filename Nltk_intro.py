
# # Introduction to NLTK
# 
# In part 1 of this assignment you will use nltk to explore the Herman Melville novel Moby Dick. Then in part 2 you will create a spelling recommender function that uses nltk to find words similar to the misspelling. 

# ## Part 1 - Analyzing Moby Dick

# In[3]:


import nltk
import pandas as pd
import numpy as np
nltk.download('punkt')

# If you would like to work with the raw text you can use 'moby_raw'
with open('moby.txt', 'r') as f:
    moby_raw = f.read()
    
# If you would like to work with the novel in nltk.Text format you can use 'text1'
moby_tokens = nltk.word_tokenize(moby_raw)
text1 = nltk.Text(moby_tokens)
#moby_raw


# ### Example 1
# 
# How many tokens (words and punctuation symbols) are in text1?
# 
# *This function should return an integer.*

# In[9]:


def example_one():
    
    return len(nltk.word_tokenize(moby_raw)) # or alternatively len(text1)

example_one()


# ### Example 2
# 
# How many unique tokens (unique words and punctuation) does text1 have?
# 
# *This function should return an integer.*

# In[ ]:


def example_two():
    
    return len(set(nltk.word_tokenize(moby_raw))) # or alternatively len(set(text1))

example_two()


# ### Example 3
# 
# After lemmatizing the verbs, how many unique tokens does text1 have?
# 
# *This function should return an integer.*

# In[ ]:


from nltk.stem import WordNetLemmatizer

def example_three():

    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(w,'v') for w in text1]

    return len(set(lemmatized))

example_three()


# ### Question 1
# 
# What is the lexical diversity of the given text input? (i.e. ratio of unique tokens to the total number of tokens)
# 
# *This function should return a float.*

# In[10]:


def answer_one():
    unique_tokens=len(set(nltk.word_tokenize(moby_raw)))
    total_tokens=len(nltk.word_tokenize(moby_raw))
    
    return unique_tokens/total_tokens

answer_one()


# ### Question 2
# 
# What percentage of tokens is 'whale'or 'Whale'?
# 
# *This function should return a float.*

# In[15]:


def answer_two():
    moby_tokens=nltk.word_tokenize(moby_raw)
    count_whale=0
    for i in moby_tokens:
        if i.lower()=='whale':
            count_whale+=1
    
    return count_whale/len(moby_tokens)

answer_two()


# ### Question 3
# 
# What are the 20 most frequently occurring (unique) tokens in the text? What is their frequency?
# 
# *This function should return a list of 20 tuples where each tuple is of the form `(token, frequency)`. The list should be sorted in descending order of frequency.*

# In[12]:


def answer_three():
    moby_tokens=nltk.word_tokenize(open('moby.txt','r').read())
    freq_dist=nltk.FreqDist(moby_tokens)
    new_dict={v:k for k,v in dict(freq_dist).items()}
    
    return [(new_dict[i],i) for i in sorted({v:k for k,v in dict(freq_dist).items()}.keys(),reverse=True)[:20]]

answer_three()


# ### Question 4
# 
# What tokens have a length of greater than 5 and frequency of more than 150?
# 
# *This function should return an alphabetically sorted list of the tokens that match the above constraints. To sort your list, use `sorted()`*

# In[13]:


def answer_four():
    moby_tokens=nltk.word_tokenize(open('moby.txt','r').read())
    freq_dist=nltk.FreqDist(moby_tokens)
    return sorted([w for w in dict(freq_dist).keys() if len(w)>5 and freq_dist[w]>150])

answer_four()


# ### Question 5
# 
# Find the longest word in text1 and that word's length.
# 
# *This function should return a tuple `(longest_word, length)`.*

# In[14]:


def answer_five():
    dict_longest={len(word):word for word in text1}
    lenght_max=max(sorted(dict_longest.keys()))
    return  (dict_longest[lenght_max],lenght_max)

answer_five()


# ### Question 6
# 
# What unique words have a frequency of more than 2000? What is their frequency?
# 
# "Hint:  you may want to use `isalpha()` to check if the token is a word and not punctuation."
# 
# *This function should return a list of tuples of the form `(frequency, word)` sorted in descending order of frequency.*

# In[15]:


def answer_six():
    uniq_w=nltk.FreqDist(nltk.word_tokenize(moby_raw))
    return sorted([(uniq_w[i],i) for i in uniq_w if i.isalpha()==True and uniq_w[i]>2000],reverse=True)

answer_six()


# ### Question 7
# 
# What is the average number of tokens per sentence?
# 
# *This function should return a float.*

# In[16]:


def answer_seven():
    
    
    return len(nltk.word_tokenize(moby_raw))/len(nltk.sent_tokenize(moby_raw))

answer_seven()


# ### Question 8
# 
# What are the 5 most frequent parts of speech in this text? What is their frequency?
# 
# *This function should return a list of tuples of the form `(part_of_speech, frequency)` sorted in descending order of frequency.*

# In[32]:


def answer_eight():
    #nltk.download('averaged_perceptron_tagger')
    words=nltk.word_tokenize(moby_raw)
    pos_=nltk.pos_tag(words)
    pos_dict={}
    for i,k in pos_:
        pos_dict[k]=pos_dict.get(k,0)+1
    
    return [(i,k)for k,i in sorted([(pos_dict[key],key) for key,values in pos_dict.items()],reverse=True)[:5]] 

answer_eight()


# ## Part 2 - Spelling Recommender
# 
# For this part of the assignment you will create three different spelling recommenders, that each take a list of misspelled words and recommends a correctly spelled word for every word in the list.
# 
# For every misspelled word, the recommender should find find the word in `correct_spellings` that has the shortest distance*, and starts with the same letter as the misspelled word, and return that word as a recommendation.
# 
# *Each of the three different recommenders will use a different distance measure (outlined below).
# 
# Each of the recommenders should provide recommendations for the three default words provided: `['cormulent', 'incendenece', 'validrate']`.

# In[ ]:


from nltk.corpus import words

correct_spellings = words.words()


# ### Question 9
# 
# For this recommender, your function should provide recommendations for the three default words provided above using the following distance metric:
# 
# **[Jaccard distance](https://en.wikipedia.org/wiki/Jaccard_index) on the trigrams of the two words.**
# 
# *This function should return a list of length three:
# `['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation']`.*

# In[9]:


def answer_nine(entries=['cormulent', 'incendenece', 'validrate']):
    from nltk.corpus import words
    from nltk.metrics.distance import jaccard_distance
    from nltk import ngrams
    nltk.download('words')
    correct_spellings = words.words()
    rec_list=[]
    for entry in entries:
        rec_l=[]
        cor_spel_new=[w for w in correct_spellings if w[0]==entry[0]]
        for i in cor_spel_new:
            ngr_i=nltk.ngrams(i,n=3)
            ngr_e=nltk.ngrams(entry,n=3)
            jac_dist=jaccard_distance(set(ngr_i),set(ngr_e))
            rec_l.append((jac_dist,i))
        rec_list.append(sorted(rec_l)[0][1])
    
    return rec_list
    
answer_nine()


# ### Question 10
# 
# For this recommender, your function should provide recommendations for the three default words provided above using the following distance metric:
# 
# **[Jaccard distance](https://en.wikipedia.org/wiki/Jaccard_index) on the 4-grams of the two words.**
# 
# *This function should return a list of length three:
# `['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation']`.*

# In[8]:


def answer_ten(entries=['cormulent', 'incendenece', 'validrate']):
    from nltk.corpus import words
    from nltk.metrics.distance import jaccard_distance
    from nltk import ngrams
    correct_spellings = words.words()
    rec_list=[]
    for entry in entries:
        rec_l=[]
        cor_spel_new=[w for w in correct_spellings if w[0]==entry[0]]
        for i in cor_spel_new:
            ngr_i=nltk.ngrams(i,n=4)
            ngr_e=nltk.ngrams(entry,n=4)
            jac_dist=jaccard_distance(set(ngr_i),set(ngr_e))
            rec_l.append((jac_dist,i))
        rec_list.append(sorted(rec_l)[0][1])
    
    return rec_list
    
answer_ten()


# ### Question 11
# 
# For this recommender, your function should provide recommendations for the three default words provided above using the following distance metric:
# 
# **[Edit distance on the two words with transpositions.](https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance)**
# 
# *This function should return a list of length three:
# `['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation']`.*

# In[23]:


def answer_eleven(entries=['cormulent', 'incendenece', 'validrate']):
    from nltk.corpus import words
    from nltk.metrics import edit_distance
    correct_spellings = words.words()
    rec_list=[]
    for entry in entries:
        rec_l=[]
        for i in correct_spellings:
            jac_dist=edit_distance(entry,i,transpositions=True)
            rec_l.append((jac_dist,i))
        rec_list.append(sorted(rec_l)[0][1])
    return rec_list
    
answer_eleven()


# In[ ]:




