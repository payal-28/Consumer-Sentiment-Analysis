# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 19:38:05 2019

@author: Payal Arora
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tweepy
from textblob import TextBlob
import csv
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB

#Sample keys
api_key1="GCYhYXmb0MoGNvNx17DvvgGhv"
api_secret1="aNGHiuKj4SsyhhyTlh8BlNZYD29bas6P1Ykw9bU6txUGp000o1"
access_token1="752547855957692416-yHtgq8FC4KbiwdnYzSzrRAaKhsyfx11"
access_token_secret1="xNT9mxJQpgik7cpcGLgq2mrHFpb39xa7qLN3zU5EU5VPe"

#establishing connection
auth=tweepy.OAuthHandler(input("Enter API key:"),input("Enter API secret:"))
auth.set_access_token(input("Enter Access Token:"),input("Enter Access Token Secret:"))
api=tweepy.API(auth)


#extracting tweets
num_of_tweets=int(input("Enter the number of tweets to be extracted:"))
tweets=tweepy.Cursor(api.search,input("Enter the shopping site of which tweets are to be extracted:"),lang="en").items(num_of_tweets)

# Open/create a file to append data to
csvFile = open('flpkrt.csv', 'a')

#Use csv writer
csvWriter = csv.writer(csvFile)

for tweet in tweets:
    csvWriter.writerow([tweet.text.encode('utf-8')])
    #print(tweet.text)   
csvFile.close()    

#reading file
flip=pd.read_csv("flpkrt.csv",header=None)
#adding header
flip.columns=["Text"]
flip.head()

#cleaning the tweets
pat1 = r'@[A-Za-z0-9]+' # this is to remove any text with @
pat2 = r'https?://[A-Za-z0-9./]+'  # this is to remove the urls
combined_pat = r'|'.join((pat1, pat2)) 
pat3 = r'[^a-zA-Z]' # to remove every other character except a-z & A-Z
combined_pat2 = r'|'.join((combined_pat,pat3)) # we combine pat1, pat2 and pat3 to pass it in the cleaning steps  
len(flip['Text'])
ps=PorterStemmer()
cleaned_tweets=[]
for i in range(0, len(flip['Text'])) :
    tweets = re.sub(combined_pat2,' ',flip['Text'][i])
    tweets = tweets.lower()
    tweets = tweets.split()
    tweets = [ps.stem(word) for word in tweets if not word in set(stopwords.words('english'))]
    tweets = ' '.join(tweets)
    cleaned_tweets.append(tweets)
    
    
flip["Cleaned_Tweets"]=cleaned_tweets
flip[["Polarity"]]= flip['Cleaned_Tweets'].apply(lambda tweet:pd.Series(TextBlob(tweet).sentiment.polarity)) 

flip["Category"]=np.where(flip["Polarity"]==0,"Neutral",np.where(flip["Polarity"]>0.00,"Positive","Negative"))

flip["Class"]=np.where(flip["Category"]=="Neutral",0,np.where(flip["Category"]=="Positive",1,-1))

#checking for null records
flip.isnull().sum()

flip['Category'].head()
flip["Cleaned_Tweets"].head()
flip["Polarity"].head()
flip["Class"].head()
flip.head()

"""
Performing sentimental analysis of flipkart tweets  
"""

def percentage(part,whole):
    return 100*float(part)/float(whole)

#calculating value counts of each category
flip['Category'].value_counts()

#storing these counts individually
positive=282
negative=625
neutral=93

#Converting into percentage        
positive=percentage(positive,num_of_tweets)
negative=percentage(negative,num_of_tweets)
neutral=percentage(neutral,num_of_tweets)
        
positive=format(positive,'.2f')
negative=format(negative,'.2f')
neutral=format(neutral,'.2f')

print("Positive percentage:",positive)
print("Negative percentage:",negative)
print("Neutral percentage:",neutral)

#checking Polarity
polarity_score=flip["Polarity"].sum()
polarity_score

#plotting pie chart       
labels=['Positive','Negative','Neutral']
values=[positive,negative,neutral]
colors=["green","yellow","red"]
patches,texts,autotexts=plt.pie(values,colors=colors,startangle=90,autopct='%.2f%%')
plt.legend(patches,labels,loc="best")
plt.title("Sentimental analysis")
plt.axis("equal")
plt.tight_layout()
plt.show()

"""
Application of classifier
"""
#creating new dataframe to store the cleaned tweets and their respective Class and save them to a csv file
tweet_sentiment=pd.DataFrame()
tweet_sentiment["cleaned_tweets"]=flip["Cleaned_Tweets"]
tweet_sentiment["class"]=flip["Class"]

tweet_sentiment.to_csv("tweet_sentiment.csv",index=False)
tweet_sentiment.shape[0]
tweet_sentiment=pd.read_csv("tweet_sentiment.csv")
tweet_sentiment.head()
tweet_sentiment.columns

#Using CountVectorizer to create a sparse matrix from the cleaned tweets 
#and defining Dependent and Independent Variables for classification

cv = CountVectorizer()
X = cv.fit_transform(tweet_sentiment['cleaned_tweets']).toarray()
y = tweet_sentiment['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#using Naive Bayes classifier
classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
score = accuracy_score(y_test, y_pred)
cm
score

#using logistic regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)
cnf_matrix =confusion_matrix(y_test, y_pred)
cnf_matrix
score = accuracy_score(y_test, y_pred)
score

