import re  #re is Python's Regular Expressions (Regex)
import string    #contains all punctuation symbols !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
import nltk
from nltk.corpus import stopwords  #the is am are was were
from nltk import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

stop_words = set(stopwords.words("english"))
lemmatizer= WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text,str):
        return ""
    
    #lowercase
    text = text.lower()

    #remove urls
    text = re.sub(r"http\S+|www\S+", " ",text)

    #remove emails
    text= re.sub(r"\S+@\S+"," ",text)
    
    #remove punct
    text= text.translate(str.maketrans("","",string.punctuation))

    #remove numbers
    text= re.sub(r"\d+"," ",text)

    #remove extra spaces
    text= re.sub(r"\s+"," ",text).strip()

    words = text.split()
    cleaned_words= [
        lemmatizer.lemmatize(word)
        for word  in words
        if word not in stop_words
    ]
    return " ".join(cleaned_words)


if __name__ == '__main__':
    sample_resume = """
    John Doe
    Email: john@example.com
    Phone: 9876543210

    Skills:
    Python, SQL, Machine Learning, TensorFlow

    Portfolio:
    https://github.com/johndoe
    """
    print(clean_text(sample_resume))
    