import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score, classification_report

from preprocess import clean_text

df = pd.read_csv(r"C:\Users\anany\OneDrive\Documents\resume classifier\data\Resume.csv")
print(df.head())
print(df.info())