import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score, classification_report

from preprocess import clean_text

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data","resume.csv")
df = pd.read_csv(DATA_PATH)

print("=" * 50)
print("Data Information")
print("="*50)

print(df.info())

print("\nFirst 5 rows:")
print(df.head())

print("\nClass Distribution:")
print(df["Category"].value_counts())

print("\nCleaning Resume Text...")
df["Resume_str"]= df["Resume_str"].apply(clean_text)
print("Cleaning complete.")

print("\nCleaned Resumes:")
print(df["Resume_str"].head())

vectorizer = TfidfVectorizer(max_features=5000)    #so we keep the 5000 most important word
x= vectorizer.fit_transform(df["Resume_str"])      #fit= Learns the vocabulary.
y= df["Category"]                                  #transform= Converts every resume into numbers. 

print("\nFeature Matrix Shape:")
print(x.shape)
print("\nTarget shape:")
print(y.shape)

