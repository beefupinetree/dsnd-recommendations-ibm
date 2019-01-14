import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import project_tests as t
import pickle
import os
os.chdir(r"C:\Users\tarek\git_workspace\dsnd-recommendations-ibm")

df = pd.read_csv('user-item-interactions.csv')
df_content = pd.read_csv('articles_community.csv')
del df['Unnamed: 0']
del df_content['Unnamed: 0']

# Show df to get an idea of the data
df.head()
