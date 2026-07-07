#This file will only read files.
#It will NOT predict.
#It will NOT clean text.

import os
from PyPDF2 import PdfReader
from docx import Document

def extract_text(file_path):
    extension= os.path.splitext(file_path)[1].lower()
    if extension == ".pdf":
        return extract_pdf(file_path)
    elif extension == ".docx":
        return extract_docx(file_path)
    else:
        raise ValueError("Unsupported file error")


def extract_pdf(file_path):
    text = ""
    reader = PdfReader(file_path)
    for page in reader.pages:
        text += page_text+"\n"
    return text

def extract_docx(file_path):
    document = Document(file_path)
    text =[]
    for paragharph in document.paragraphs:
        text.append(paragraph.text)
    return "\n".join(text)
