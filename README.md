Table of contents:
● General Info
● Requirement
General Info
This project is a simple implementation of K Nearest Neighbour classifier and it could be used
for sentiment analysis of movie reviews.
The answer function is what implements the KNN classifier. It could be found in the lab1.py file.
Requirement
This project is using 3.7.4 and for running it you could use your preferred IDE or Jupyter
Notebook.
It also uses a number of libraries that are listed below:
Numpy, Re, Bs4, Sklearn, Nltk
These are the libraries need to be installed on the virtual env or your local computer in order to
run this program.
On mac and linux:
python3 -m pip install "SomeProject"
On windows:
py -m pip install "SomeProject"
Creating a virtual environment on your machine and installing these libraries are preferable.
After installing needed libraries this could help you import required packages:
import numpy as np
import re
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
