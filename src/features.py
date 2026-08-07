from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

#Split the data
X=df['clean_message']
y=df['label']


