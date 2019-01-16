import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
os.chdir(r"C:\Users\tarek\git_workspace\dsnd-recommendations-ibm")
import project_tests as t

df = pd.read_csv('user-item-interactions.csv')
df_content = pd.read_csv('articles_community.csv')
del df['Unnamed: 0']
del df_content['Unnamed: 0']

# Show df to get an idea of the data
df.head()

# Fill in the median and maximum number of user_article interactios below
interactions_user = df.groupby('email')['article_id'].nunique()
interactions_user = interactions_user.sort_values(ascending=False)
print(interactions_user.quantile(.5))
print(interactions_user.max())

median_val = 3  # 50% of individuals interact with ____ number of articles or fewer.
max_views_by_user = 135  # The maximum number of user-article interactions by any 1 user is ______. 

# Find and explore duplicate articles
dups = df[df.duplicated(keep=False)].sort_values(by=['article_id'])
dups_count = dups.groupby('article_id')['email'].nunique()

# Remove any rows that have the same article_id - only keep the first
df = df.sort_values(by=['article_id'])
df_no_dups = df.drop_duplicates(subset=['article_id'], keep='first',
                                inplace=False).sort_values(by=['article_id'])
