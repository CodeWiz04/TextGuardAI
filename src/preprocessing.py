import pandas as pd

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


