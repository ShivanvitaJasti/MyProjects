#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import needed libraries
# needed for directory access
import os 
import nltk
import pandas as pd
import nltk.corpus
import matplotlib.pyplot as plot
from nltk.probability import FreqDist

# get and set your working directory
os.getcwd()
os.chdir('/Users/Public')
os.getcwd()


# In[3]:


# Article - 1: Israel leverages Russia ties to try to mediate between West and Putin. Will it work?
# reading the text file

textfile_1 = open('Article_1.txt', encoding='utf8')
Article_1 = textfile_1.read()
print(Article_1)


# In[4]:


from nltk.tokenize import RegexpTokenizer
tokenizer_1 = RegexpTokenizer(r'\w+')
token_results_1 = tokenizer_1.tokenize(Article_1)
print(token_results_1)


# In[5]:


# number of tokens

len(token_results_1)


# In[6]:


# calculating the frequency of words in the tokens, and displaying top 10 words

FreqDist_before = FreqDist()

for a in token_results_1:
    FreqDist_before[a] = FreqDist_before[a] + 1

FreqDist_before_top10 = FreqDist_before.most_common(10)
FreqDist_before_top10   


# In[26]:


#displaying a few sentences based on the keywords
sentence = nltk.sent_tokenize(Article_1)

for a in sentence:
 if "Russia" in a:
  print(a)
for b in sentence:
 if "Ukraine" in b:
  print(b)


# In[7]:


# Removing smaller words, punctuations, numbers and stopwords from the tokens

nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize

Tokenized_words_1 = nltk.word_tokenize(Article_1)

Tokenized_words_1 = [word.lower() for word in Tokenized_words_1 if word.isalpha()]

# Removing smaller-character tokens (mostly punctuation)
cleaned_tokens_1 = [word for word in Tokenized_words_1 if len(word) > 3]

# Removing numbers
cleaned_tokens_1 = [word for word in Tokenized_words_1 if not word.isnumeric()]

cleaned_tokens_1 = [word for word in Tokenized_words_1 if not word in stopwords.words()]

print(cleaned_tokens_1)


# In[8]:


# length of cleaned tokens

len(cleaned_tokens_1)


# In[9]:


# calculating the frequency of words in the cleaned tokens, and displaying top 10 words

FreqDist_after = FreqDist()
for b in cleaned_tokens_1:
    FreqDist_after[b] = FreqDist_after[b] + 1
    
FreqDist_after_top10 = FreqDist_after.most_common(10)
FreqDist_after_top10  


# In[10]:


# plotting the frequency of cleaned tokens

plot_words_1 = nltk.FreqDist(cleaned_tokens_1)
plot_words_1.plot(20);


# In[11]:


# Article - 2: Russia Is Sending Mercenaries and Syrians to Ukraine, Western Officials Say
# reading the text file

textfile_2 = open('Article_2.txt', encoding='utf8')
Article_2 = textfile_2.read()
print(Article_2)


# In[12]:


from nltk.tokenize import RegexpTokenizer
tokenizer_2 = RegexpTokenizer(r'\w+')
token_results_2 = tokenizer_2.tokenize(Article_2)
print(token_results_2)


# In[13]:


# number of tokens

len(token_results_2)


# In[15]:


# calculating the frequency of words in the tokens, and displaying top 10 words

FreqDist_before = FreqDist()

for b in token_results_2:
    FreqDist_before[b] = FreqDist_before[b] + 1

FreqDist_before_top10 = FreqDist_before.most_common(10)
FreqDist_before_top10   


# In[27]:


#displaying a few sentences based on the keywords
sentence = nltk.sent_tokenize(Article_2)

for a in sentence:
 if "Russia" in a:
  print(a)
for b in sentence:
 if "Ukraine" in b:
  print(b)


# In[16]:


# Removing smaller words, punctuations, numbers and stopwords from the tokens

Tokenized_words_2 = nltk.word_tokenize(Article_2)

Tokenized_words_2 = [word.lower() for word in Tokenized_words_2 if word.isalpha()]

# Removing smaller-character tokens (mostly punctuation)
cleaned_tokens_2 = [word for word in Tokenized_words_2 if len(word) > 3]

# Removing numbers
cleaned_tokens_2 = [word for word in Tokenized_words_2 if not word.isnumeric()]

cleaned_tokens_2 = [word for word in Tokenized_words_2 if not word in stopwords.words()]

print(cleaned_tokens_2)


# In[17]:


# length of cleaned tokens

len(cleaned_tokens_2)


# In[18]:


# calculating the frequency of words in the cleaned tokens, and displaying top 10 words

FreqDist_after = FreqDist()
for b in cleaned_tokens_2:
    FreqDist_after[b] = FreqDist_after[b] + 1
    
FreqDist_after_top10 = FreqDist_after.most_common(10)
FreqDist_after_top10  


# In[19]:


# plotting the frequency of cleaned tokens

plot_words_2 = nltk.FreqDist(cleaned_tokens_2)
plot_words_2.plot(20);

