from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

#Split the data
X=df['clean_message']
y=df['label']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

tfidf=TfidfVectorizer(max_features=5000)     #Collects unique words, find count occurences of each word,form it into vector called bag of words
                                             #but bag of words isn't enough as all the words get same importance, so we use tfidf which gives weightage to words based on their importance
                                             #Tf=>how important is this word inside the document(Number of times word appears/Total words in document)
                                             #idf=>gives more importance to rare words(log(Total Documents/Number of documents containing the word))
