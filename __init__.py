import numpy as np, pandas as pd, requests, re, unicodedata, pymongo
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity