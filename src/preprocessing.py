import pandas as pd
import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
df=pd.read_csv("data/raw/spam.csv",encoding='latin-1')   #As dataset contains some special characters it would throw UnicodeDecodeError
print(df.head())

#Rename columns
df=df.rename(
    columns={
        "v1":"label",
        "v2":"message"
    }
)

df=df.loc[:,["label","message"]]

print("Shape:",df.shape)

print(df["label"].value_counts())

#character count
df['character_count']=df['message'].str.len()

#word count
df['word_count']=df['message'].str.split().str.len()

#Summary
print(df["character_count"].describe())
print(df["word_count"].describe())


#Creating reusable objects
stop_words=set(stopwords.words("english"))
lemmatizer=WordNetLemmatizer()   #lexical database giving base form,synonyms,POS etc

def clean_text(message):
    #Lowercase
    message=message.lower()

    #Remove Punctuation
    message=message.translate(str.maketrans("","",string.punctuation))
    #Tokenize
    tokens=word_tokenize(message)
    #Remove stop_words and lemmatize
    cleaned_tokens=[
        lemmatizer.lemmatize(token) for token in tokens if token not in stop_words
    ]
    #join back into a sentence
    return " ".join(cleaned_tokens)
    


