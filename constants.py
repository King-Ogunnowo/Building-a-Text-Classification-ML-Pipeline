"""
Constants file to hold all programming dependencies for FiTCAM model pipeline

Author: Oluwaseyi E. Ogunnowo
Date: 8th July 2022.
"""

# --- importing all dependencies
import nltk
import logging
from nltk.corpus import stopwords

# --- instantiating logging object
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# --- stopwords for text cleaning
nltk.download('stopwords')
stop_words = stopwords.words('english')

