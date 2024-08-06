# UTILS for the Danish HCA SA study

import os

import json
import sklearn
import pandas as pd
from importlib import reload
from scipy import stats
import numpy as np
from scipy.stats import norm
from scipy.stats import spearmanr


import nltk
#nltk.download('punkt')

# SA
import asent
import spacy
from afinn import Afinn
from sentida import Sentida

import random

# for SA with transformers
from transformers import pipeline

import sklearn
import re
from scipy.stats import norm
import xlrd

# plot
import seaborn as sns
import matplotlib.pyplot as plt

# to use custom plotting functions (modified from figs.py)
# import sys
# sys.path.append('/Users/au324704/Library/Mobile Documents/com~apple~CloudDocs/CHCAA/FABULANET/')
import saffine
import saffine.multi_detrending as md
import saffine.detrending_method as dm

# for inter rater reliability
from statsmodels.stats import inter_rater as irr
import krippendorff as kd


# Roberta
#!pip install sentencepiece
#!pip install protobuf
from transformers import pipeline