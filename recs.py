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
max_views_by_user = df.groupby('email')['article_id'].count().max()  # The maximum number of user-article interactions by any 1 user is ______. 

# Find and explore duplicate articles
dups = df[df.duplicated(keep=False)].sort_values(by=['article_id'])
dups_count = dups.groupby('article_id')['email'].nunique()

# Remove any rows that have the same article_id - only keep the first
df = df.sort_values(by=['article_id'])
df_no_dups = df.drop_duplicates(subset=['article_id'], keep='first',
                                inplace=False).sort_values(by=['article_id'])

interactions_article = df.groupby('article_id')['email'].count()

unique_articles = interactions_article[interactions_article>0].count()  # The number of unique articles that have at least one interaction
total_articles = df_content['article_id'].nunique()  # The number of unique articles on the IBM platform
unique_users = df[['email']].nunique().max()  # The number of unique users
user_article_interactions = len(df) # The number of user-article interactions

# The most viewed article in the dataset as a string with one value following
# the decimal 
most_viewed_article_id = interactions_article[interactions_article ==
                                              interactions_article.max()].index[0].astype(str)
# The most viewed article in the dataset was viewed how many times?
max_views = interactions_article.max()

# No need to change the code here - this will be helpful for later parts of
# the notebook. Run this cell to map the user email to a user_id column and
# remove the email column


def email_mapper():
    coded_dict = dict()
    cter = 1
    email_encoded = []

    for val in df['email']:
        if val not in coded_dict:
            coded_dict[val] = cter
            cter += 1

        email_encoded.append(coded_dict[val])
    return email_encoded


email_encoded = email_mapper()
del df['email']
df['user_id'] = email_encoded

# show header
df.head()

## If you stored all your results in the variable names above, 
## you shouldn't need to change anything in this cell

sol_1_dict = {
    '`50% of individuals have _____ or fewer interactions.`': median_val,
    '`The total number of user-article interactions in the dataset is ______.`': user_article_interactions,
    '`The maximum number of user-article interactions by any 1 user is ______.`': max_views_by_user,
    '`The most viewed article in the dataset was viewed _____ times.`': max_views,
    '`The article_id of the most viewed article is ______.`': most_viewed_article_id,
    '`The number of unique articles that have at least 1 rating ______.`': unique_articles,
    '`The number of unique users in the dataset is ______`': unique_users,
    '`The number of unique articles on the IBM platform`': total_articles
}

# Test your dictionary against the solution
t.sol_1_test(sol_1_dict)


def get_top_articles(n, df=df):
    '''
    INPUT:
    n - (int) the number of top articles to return
    df - (pandas dataframe) df as defined at the top of the notebook

    OUTPUT:
    top_articles - (list) A list of the top 'n' article titles
    '''
    interactions_article = df.groupby('article_id')['user_id'].count().sort_values(ascending=False)
    top_n = interactions_article.index[:n]
    top_articles = list(set(df.loc[df['article_id'].isin(top_n)]['title']))
    return top_articles # Return the top article titles from df (not df_content)

def get_top_article_ids(n, df=df):
    '''
    INPUT:
    n - (int) the number of top articles to return
    df - (pandas dataframe) df as defined at the top of the notebook

    OUTPUT:
    top_articles - (list) A list of the top 'n' article titles
    '''
    top_articles = list(interactions_article.index[:n])

    return top_articles # Return the top article ids

print(get_top_articles(10))
print(get_top_article_ids(10))

# Test your function by returning the top 5, 10, and 20 articles
top_5 = get_top_articles(5)
top_10 = get_top_articles(10)
top_20 = get_top_articles(20)

# Test each of your three lists from above
t.sol_2_test(get_top_articles)
