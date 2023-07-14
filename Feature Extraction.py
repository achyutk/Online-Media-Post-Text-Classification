# Importing Necessary Packages
import pandas as pd
from googletrans import Translator
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('stopwords')
nltk.download('punkt')



"""-------------------------------------------------------------------------------------------------------------------------------------""""
"""## Read and Write a file"""

# Code to write a dataframe
def write_file(df,path,name):
    df.to_csv(path+"/"+name+".csv")

# Code to read dataframe
def read_file(path,name):
    dataframe=pd.read_csv(path + "/" + name,delimiter = "\t")
    return dataframe



"""-------------------------------------------------------------------------------------------------------------------------------------"""
"""Reading Files"""

path='C:/Southampton Studies/ML Tech/Coursework/assignment-comp3222-comp6246-mediaeval2015-dataset/assignment-comp3222-comp6246-mediaeval2015-dataset' # Path of the project
train_filename="mediaeval-2015-trainingset.txt" # Path to training set. Generally kept in the same file as the code for easy processing
test_filename="mediaeval-2015-testset.txt" # Path to testing set.  Generally kept in the same file as the code for easy processing


df_test=read_file(path,test_filename) #Reading Test.txt file to convert to csv file
write_file(df_test,path,"test_data") #Writing Test file to convert to csv file

df_train=read_file(path,train_filename) #Reading Train.txt file to convert to csv file
write_file(df_train,path,"training_data") #Writing Test file to convert to csv file




"""-------------------------------------------------------------------------------------------------------------------------------------"""
"""Execute the code below to perform EDA"""

#The function below performs basic EDA like head of each dataframe, distribution of each labels,etc
def eda(df):
    print(df.head())
    print(df.info())

    for i in df.columns:
        user= set(df[i])
        print(i," : ",len(user) )

    print("Labels distribution: ", df['label'].value_counts())

eda(df_train) # Calling EDA function




"""-------------------------------------------------------------------------------------------------------------------------------------"""
"""Preprocessing"""

#Splitting the tweet content and the link
def splitt (dataframe):
    dataframe['link']=dataframe.tweetText.str.split(" http:",n=1,expand=True)[1]
    dataframe['link'] = "http:" + dataframe['link'].map(str)
    dataframe['tweetText']=dataframe.tweetText.str.split(" http:",n=1,expand=True)[0]
    return dataframe

df_train = splitt(df_train) 
df_test = splitt(df_test)


"""# Identifying the Languages"""

# Function that identifies language of a the tweets.
def language_detection(data):
    translator = Translator() # Creating object of class Translator from googletrans library
    langDetected=[]
    for i in data:
        detection = translator.detect(i)
        langDetected.append(detection.lang)
    return langDetected

df_test['language']=language_detection(df_test['tweetText']) #Calling function to identify language for test set
df_train['language']=language_detection(df_train['tweetText']) #Calling function to identify language for training set
write_file(df_test,path,"test_data_identified") # Writing the df to csv file. Used if needed 
write_file(df_train,path,"training_data_identified") # Writing the df to csv file. Used if needed 


"""##### Translating non-english tweets to english"""

#Uncomment the code below if you want to load dataset when laguage is already indetified
# df_train = read_file(path,"train_data_identified")
# df_test = read_file(path,"test_data_identified")

#Function translates the tweet if the language is not equal to english 
def translate(data,language):
    translator = Translator() # Creating object of class Translator from googletrans library
    for i in range(len(data)):
        if language[i]!='en':
            translations = translator.translate(data[i])
            data[i]=translations
    return data

df_train.tweetText=translate(df_train['tweetText'],df_train['language']) # Calling the translate function to translate non english dataset for training set.
df_test.tweetText=translate(df_test['tweetText'],df_test['language']) # Calling the translate function to translate non english dataset for testing set.
write_file(df_test,path,"test_data_translated") # Writing the df to csv file. Used if needed 
write_file(df_train,path,"training_data_translated") # Writing the df to csv file. Used if needed 


"""#####Additional Feature Extraction + Data Transformation"""

#Uncomment the code below if you want to load dataset when laguage is already indetified
# df_train = read_file(path,"training_data_translated")
# df_test = read_file(path,"test_data_translated")


#The function below has all different feature extractions
def featureextraction(dataframe,neg_words,pos_words,unclassified):

    #Converting labels to 0 or 1
    dataframe['label']=dataframe['label'].apply(lambda x:1 if x in ["fake","humor"] else 0)

    #Calculating Tweetlength of tweets
    dataframe['tweetlength'] = dataframe['tweetText'].apply(lambda x:len(x))

    #counting number of words in tweet
    dataframe['wordcount']=dataframe['tweetText'].apply(lambda x:len(x.split()))

    # Checking for exclaimation in tweets
    dataframe['containsExclaimation'] = dataframe['tweetText'].apply(lambda x: 1 if "!" in x else 0)

    # Checking for question mark in tweets
    dataframe['containsQuestion'] = dataframe['tweetText'].apply(lambda x: 1 if "?" in x else 0)

    #Count questions in tweets
    dataframe['countQuestions'] = dataframe['tweetText'].apply(lambda x: x.count("?"))

    #Count questions in tweets
    dataframe['countHashtags'] = dataframe['tweetText'].apply(lambda x: x.count("#"))

    #Count mentions in tweets
    dataframe['countmentions'] = dataframe['tweetText'].apply(lambda x: x.count("@"))

    # Filtering stopwords from the tweets
    dataframe['filteredtweets']=dataframe['tweetText'].apply(lambda x: wordfilter(x))

    #Filtering words with less than length less than 1
    dataframe['filteredtweets']=dataframe['filteredtweets'].apply(lambda x: [i for i in x if len(i)>1])

    #Finding positive words in tweets
    dataframe['numpositivewords']=dataframe['filteredtweets'].apply(lambda x: len([i for i in x if i in pos_words]))

    #Finding negative words in tweets
    dataframe['numnegativewords']=dataframe['filteredtweets'].apply(lambda x: len([i for i in x if i in neg_words]))

    #Finding negative words in tweets
    dataframe['numneutralwords']=dataframe['filteredtweets'].apply(lambda x: len([i for i in x if i not in unclassified]))

    #Parts of speech tagging
    dataframe['taggedData']=dataframe['filteredtweets'].apply(lambda x: nltk.pos_tag(x))


    #Comprehensive lost tags used to tag the words of tweets
    pos_tag_list=["CC","CD","DT","EX","FW","IN","JJ","JJR","JJS","LS","MD","NN","NNS","NNP","NNPS","PDT","POS","PRP","PRP","RB","RBR","RBS","RP","SYM","TO","UH","VB","VBD","VBG","VBN","VBP","VBZ","WDT","WP","WP$","WRB"]

    #Converting tags to column and checking if it exists in the dataset
    for i in pos_tag_list:
        dataframe[i]=dataframe['taggedData'].apply(lambda x:1 if len([j[0] for j in x if j[1]==i])>0 else 0)

    return dataframe

#Function to remove stopwords from a sentence
def wordfilter(tweet):
    word_tokens = word_tokenize(tweet)
    stop_words = set(stopwords.words('english'))
    filtered_sentence = []
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)
    return filtered_sentence

#Reading files with negative word
file = open(path+'/negative-words.txt', 'r')
neg_words = file.read().split()
#Reading files with postive word
file = open(path+'/positive-words.txt', 'r')
pos_words = file.read().split()
unclassified = neg_words + pos_words #List with all negative and postive words



df_train = featureextraction(df_train,neg_words,pos_words,unclassified)
df_test = featureextraction(df_train,neg_words,pos_words,unclassified)
write_file(df_test,path,"test_data_extracted") # Writing the df to csv file. Used if needed 
write_file(df_train,path,"training_data_extracted") # Writing the df to csv file. Used if needed 