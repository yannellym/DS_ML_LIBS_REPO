#!/usr/bin/env python
# coding: utf-8

# In[38]:


import pandas as pd
import numpy as np


# In[39]:


# reads the dataset and creates the df variable as a reference
df = pd.read_csv('Reviews.csv')


# In[40]:


# let's take a look at the first rows of our dataset
df.head()


# In[41]:


# data we will be using: Text (complete product review information),
# Summary (summary of the entire review), Score (product rating provided by the customer).


# In[42]:


# imports the necessary libraries in order to visualize our data
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.express as px


# In[43]:


# takes a look at the variable "Score" to see if majority of the customer ratings are positive or negative
# creates the histogram
fig = px.histogram(df, x="Score")
# edits the features of the histogra,
fig.update_traces(marker_color="magenta", marker_line_color='rgb(8,4,8,107)', marker_line_width=1.5)
# sets the title of the histogram
fig.update_layout(title_text='Product Score')
fig.show()


# In[44]:


# based on the above, we can see that almost all of the ratings are positive. 
# This MIGHT mean that almost all of the reviews are also positive too


# In[45]:


# Now we're going to create a word cloud to show us the most used words in the reviews
# we need to download wordcloud
# https://www.datacamp.com/tutorial/wordcloud-python
get_ipython().system('pip install wordcloud')


# In[46]:


# This didn't work for me but let's leave it here for reference

# Let's create some word clouds to see what words are frequently used in the reviews

#https://www.nltk.org/. (NATURAL LANGUAGE TOOLKIT) -> can let you tokenize, tag text, identify named entities,
# parse trees, etc
# import nltk
# nltk.download('stopwords')

# Text may contain stop words like ‘the’, ‘is’, ‘are’. Stop words can be filtered from the text to be processed. 
# There is no universal list of stop words in nlp research, however the nltk module contains a list of stop words.
# https://pythonspot.com/nltk-stop-words/
# from nltk.corpus import stopwords


# In[47]:


# # first, we need to create a stopword list
# stopwordsV = set(stopwords)
# stopwordsV.update(["br", "href"])
# textt = " ".join(review for review in df.Text)
# wordcloud = WordCloud(stopwords=stopwordsV).generate(textt)
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")
# # saves the image as a png
# plt.savefig('wordcloud11.png')
# plt.show()


# In[48]:


from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


# In[49]:


# we're first going to start with 1 review and see what it shows us!

# first review in the dataset
text = df['Text'][0]
# Create and generate a word cloud image:
wordcloud = WordCloud().generate(text)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[50]:


# lower max_font_size, change the maximum number of word and lighten the background:
wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
plt.figure()
# the argument interpolation="bilinear" in the plt.imshow(). This is to make the displayed image appear more smoothly.
# https://matplotlib.org/stable/gallery/images_contours_and_fields/interpolation_methods.html
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# In[51]:


alltext = " ".join(review for review in df['Text'])
print ("There are {} words in the combination of all review.".format(len(alltext)))


# In[52]:


# Create stopword list:

# Text may contain stop words like ‘the’, ‘is’, ‘are’. Stop words can be filtered from the text to be processed. 
# There is no universal list of stop words in nlp research, however the nltk module contains a list of stop words.
# https://pythonspot.com/nltk-stop-words/

stopwords = set(STOPWORDS)
# adding to our stop words since the words below are mentioned and don't add value to our data
stopwords.update(["product","products", 'bought', 'looks', 'found', 'several', 'food', 'smells'])

# Generate a word cloud image
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)

# Display the generated image:
# the matplotlib way:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[53]:


# We can see that the most mentioned words 'better', 'vitality', 'good', 'quality'. These words are often 
# associated with positive feelings. However, with such a range of reviews, we will need to do more analysis
# in order to find out more information


# # Classify reviews into "positive" and "negative" 

# In[54]:


# Next Step: Classify reviews into "positive" and "negative" to use this training data for our sentiment classification model.
# positive reviews are +1 
# negative reviews are - 1
# If a review has a 'Score' of < 3, it will be classified as -1, if the score equals 3, it will be dropped as it is neutral
# This will help us obtain positive and negative reviews.


# In[55]:


# assign reviews with score > 3 as positive sentiment
# score < 3 negative sentiment
# remove score = 3

# create a new df with only the scores from our data that do not equal 3
df = df[df['Score'] != 3]
# create a new column named "Sentiment" that includes the following data:
# scores with a sentiment of +1 if the rating > 3 else -1 
df['Sentiment'] = df['Score'].apply(lambda rating: +1 if rating > 3 else -1)


# In[56]:


# lets take a look to see how our sentiment column populated
df


# # Data analysis 

# In[57]:


# Lets create two data frams (one with all the positives reviews, and another with all the negative reviews)

positivedf = df[df['Sentiment'] == 1]
negativedf = df[df['Sentiment'] == -1]


# In[58]:


# next, let's create a word cloud to see the positive words used 

#get the stop words and put them in a set
stopwords = set(STOPWORDS)
# update the stopwords if there are any words you might need
stopwords.update(["one","good","great", "stuff", "dog", "cup", "really", "product", "way"]) 
# these words are being removed because they do not add to our analysis
# for every review that is positive, get its summary, and join them together
pos = " ".join(review for review in positivedf.Summary)
# create a cloud that takes into consideration the stopwords and generates info from the pos
wordcloud2 = WordCloud(stopwords=stopwords).generate(pos)
# set the interpolation to bilinear
plt.imshow(wordcloud2, interpolation='bilinear')
# removes the axes
plt.axis("off")
plt.show()


# In[59]:


# looks like most positive reviews contain the following words: 
# "love","best", "taste", "delicious", "tasty", "healthy", "price", "awesome", "treat", "Excellent", "flavor"


# In[60]:


# for every review that is positive, get its summary, and join them together
neg = " ".join(str(review) for review in negativedf.Summary)
# create a cloud that takes into consideration the stopwords and generates info from the pos
wordcloud3 = WordCloud(stopwords=stopwords).generate(neg)
# set the interpolation to bilinear
plt.imshow(wordcloud3, interpolation='bilinear')
# removes the axes
plt.axis("off")
plt.show()


# In[61]:


# looks like most negative reviews contain the following words: 
# "dissapointed", "flavor", "price", "terrible", "taste", "money", "awful", "bad", "yuck"


# # Visualize the sentiment distribution

# In[62]:


# let's create a column named "sentimentRS' that represents the sentiment review's sumamry

# sentimentRS columm will now equal to the sentiment column that had all -1 values replaced with the word "negative"
df['sentimentRS'] = df['Sentiment'].replace({-1 : 'negative'})
# sentimentRS columm will now equal to the sentiment column that had all 1 values replaced with the word "positive"
df['sentimentRS'] = df['sentimentRS'].replace({1 : 'positive'})


# take a look at our update column!! 
df[['Sentiment', 'sentimentRS']]


# In[63]:


# next, let's make a histogram of our sentimentRS column to see how our reviews compare

# create a fig variable and call plotly's px.histogram to refer to our data, and plot the "sentimentRS" column
fig = px.histogram(df, x="sentimentRS")
# let's give it some style
fig.update_traces(marker_color="mediumaquamarine",marker_line_color='rgb(8,48,107)',marker_line_width=2)
fig.update_layout(title_text='Product Sentiment')
fig.show()


# In[64]:


# as we can see from our graph, there are about 443.777k positive reviews, and 82.037k negative reviews


# # Build the model

# In[65]:


# our sentiment analysis model with take the reviews as input, and then come up with a
# prediction as to wether the review is positive or negative.

# since the task is classification, we will train a logisitc regression model to do it.


# In[66]:


# step 1: clean the data

# we will be removing all the punctuation from our data
# let's create a function that returns the words that do not match our set of punctuations
def remove_punctuation(text):
    final = " ".join(word for word in text if word not in(".", "?", ":", ";", "!", "''"))
    return final


# In[94]:


# to the column "Text", apply our remove_punctuation function
df['Text'] = df['Text'].apply(remove_punctuation)
# from our column "Summary", drop/remove any rows that are NaN 
df = df.dropna(subset=['Summary'])
df = df.dropna(subset=['Text'])
# the summary column will equal the summary column after it has been cleaned up by the remove_punctuation column
# We use .loc to access the 'Summary' column and apply the remove_punctuation function to it. 
# Note: The : inside .loc[:, 'Summary'] specifies that we want to select all rows (:) and the 'Summary' column.
df.loc[:, 'Summary'] = df['Summary'].apply(remove_punctuation)


# In[95]:


# step 2: splitting the data frame
# lets take a look at our new data

# create a newdf out of the summary and sentiment columns
newdf = df[['Summary', 'Sentiment']]
newdf


# # split the data into train and test sets: 80-20 split

# In[128]:


# random split train and test data

# create a new var "index", and assign it to the index of the dataframe df. 
# The index is the unique identifier for every row in df.
index = newdf.index
# adds a new column to the DataFrame df called 'random_number' and fills it with random numbers generated 
# from a normal distribution using NumPy's np.random.randn function. The length of the new column is set
# to be the same as the length of the index of the DataFrame.
newdf['random_number'] = np.random.randn(len(index))

#  creates a new DataFrame train by selecting only the rows in the original DataFrame df where the value 
# in the 'random_number' column is less than or equal to 0.8. This creates a random sample of 80% of the 
# original data, which will be used for training a machine learning model.
train = newdf[newdf['random_number'] <= 0.8]

# creates a new DataFrame test by selecting only the rows in the original DataFrame df where the value 
# in the 'random_number' column is greater than 0.8. This creates a random sample of 20% of the original 
# data, which will be used for testing the machine learning model.
test = newdf[newdf['random_number'] > 0.8]
train


# # Let's create our bag of words

# In[129]:


# we will use the Scikit-learn library to utilize the count vectorizer
# we will transform the text in our df into a bag of words model. It will convert to a sparse matrix of integers. 
# the num of of occurrences of each word will be counted and printed out
# It essentially takes a set of text documents and creates a vocabulary of unique words, then assigns a count to each word in each document.
# ** this is needed for the logistic regression algo to understand the text


# In[130]:


# lets install scikit learn
get_ipython().system('pip install scikit-learn')


# In[131]:


# imports the count vectorizer
from sklearn.feature_extraction.text import CountVectorizer

#specifies a regular expression pattern to identify tokens. This pattern matches any sequence of one 
# or more word characters (letters, digits, or underscores) bounded by word boundaries (\b).
vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')

#fits the vectorizer on the training data and transforms it into a document-term matrix.
# The resulting train_matrix is a sparse matrix where each row corresponds to a document and each column 
# corresponds to a word in the vocabulary. The value in each cell represents the frequency of that word in that document.
train_matrix = vectorizer.fit_transform(train['Summary'])

# applies the same vectorization process to the test data. However, instead of fitting the vectorizer to 
# the data again, it uses the vocabulary and word counts from the training data to transform the test data
# into a document-term matrix. The resulting test_matrix is also a sparse matrix with the same dimensions 
# as train_matrix.
test_matrix = vectorizer.transform(test['Summary'])


# # Split the target and independent variables

# In[151]:


# lets import our logistic regression 
from sklearn.linear_model import LogisticRegression

# create var to refer to initiate our LR class
lr = LogisticRegression(max_iter = 10000)


# In[152]:


# name our variables accordingly

# x values that will train and test on our training data (80% of the data)
X_train = train_matrix
X_test = test_matrix

# y values that will train on the remaining 20% data (sentiment column of the df and test on the sentiment value of the df)
y_train = train['Sentiment']
y_test = test['Sentiment']


# # Fit the model on the data

# In[153]:


lr.fit(X_train, y_train)
# I had to go back and adjust the number of iterations since our model wasn't converging.
# First I tried 500, then 1,000, then 5,000


# In[154]:


# let's make a prediction
predictions = lr.predict(X_test)
predictions


# # Test the model

# In[155]:


# let's test how accurate the model is

# first, let's import some metrics from sklearn
from sklearn.metrics import confusion_matrix,classification_report


# In[156]:


# creates a new arry based on our y_test data
new = np.asarray(y_test)
# creates a confusion matrix based on predictions and ytest data
confusion_matrix(predictions,y_test)


# In[157]:


# finds the accuracy, precision, recall:

print(classification_report(predictions,y_test))


# In[158]:


# it seems as our positive data had more positive reviews than negatives.
# this lead to the overall precision of 84% 
# we have a recall of .84 for the positive ones, and .38 for the negative ones. 
# We also have an f1 score of 91 for positives and .03 for negatives.

# based on these findings, I should have cleaned up the data some more to make sure there wasn't such a skewed number of 
# reviews. This would have helped our model learn a little more about the negative reviews. Which in turn
# could have improved the accuracy, precision, and fscore of the negative reviews prediction


# In[ ]:




