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

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2, random_state=42)
print("\nTraining sample", x_train.shape[0])
print("\nTesting sample", x_test.shape[0])

model = LogisticRegression(max_iter=1000)   #later Naive-Bayes Linear-SVM Random Fores
print("\nTraining the model...")
model.fit(x_train, y_train)
print("Training complete.")

print("\nEvaluating the model...")
y_pred = model.predict(x_test)
print("Evaluation complete.")

print("Accuracy:")
accuracy = accuracy_score(y_test,y_pred)
print(f"\nModel Accuracy:{accuracy *100:.2f}%")

print("Classification Report:")
print(classification_report(y_test,y_pred))

MODEL_DIR= os.path.join(BASE_DIR,"saved_model")
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(model, os.path.join(MODEL_DIR,"model.pkl"))
joblib.dump(vectorizer,os.path.join(MODEL_DIR,"vectorizer.pkl"))
print(f"\nModel saved successfully!")
