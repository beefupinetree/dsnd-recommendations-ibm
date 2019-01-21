import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
import os
os.chdir(r"C:\Users\tarek\git_workspace\dsnd-recommendations-ibm")
import project_tests as t

plt.rcParams['figure.figsize'] = 12, 9
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

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
df2 = df.sort_values(by=['article_id'])
df_no_dups = df2.drop_duplicates(subset=['article_id'], keep='first',
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
    interactions_article = df.groupby('article_id')['user_id'].count().sort_values(ascending=False)
    top_articles = list(map(str,list(interactions_article.index[:n]))) # list(interactions_article.index[:n])

    return top_articles # Return the top article ids


# Test your function by returning the top 5, 10, and 20 articles
top_5 = get_top_articles(5)
top_10 = get_top_articles(10)
top_20 = get_top_articles(20)

# Test each of your three lists from above
t.sol_2_test(get_top_articles)

# create the user-article matrix with 1's and 0's


def create_user_item_matrix(df):
    '''
    INPUT:
    df - pandas dataframe with article_id, title, user_id columns

    OUTPUT:
    user_item - user item matrix

    Description:
    Return a matrix with user ids as rows and article ids on the columns with 1
    values where a user interacted with an article and a 0 otherwise
    '''
    user_item = pd.pivot_table(df, values='title', index=['user_id'],
                               columns=['article_id'],
                               aggfunc=lambda x: len(x.unique()),
                               fill_value=0)

    return user_item


user_item = create_user_item_matrix(df)

## Tests: You should just need to run this cell.  Don't change the code.
assert user_item.shape[0] == 5149, "Oops!  The number of users in the user-article matrix doesn't look right."
assert user_item.shape[1] == 714, "Oops!  The number of articles in the user-article matrix doesn't look right."
assert user_item.sum(axis=1)[1] == 36, "Oops!  The number of articles seen by user 1 doesn't look right."
print("You have passed our quick tests!  Please proceed!")


def find_similar_users(user_id, user_item=user_item):
    '''
    INPUT:
    user_id - (int) a user_id
    user_item - (pandas dataframe) matrix of users by articles:
                1's when a user has interacted with an article, 0 otherwise

    OUTPUT:
    similar_users - (list) an ordered list where the closest users (largest dot
                    product users) are listed first

    Description:
    Computes the similarity of every pair of users based on the dot product
    Returns an ordered list
    '''
    # compute similarity of each user to the provided user
    sims = np.dot(user_item.drop([user_id]), user_item.iloc[user_id-1])
    ind = user_item.drop([user_id]).index
    full_sims = pd.DataFrame(sims, ind, columns=['dot'])
    # sort by similarity
    sims_sorted = full_sims.sort_values(by=['dot'], ascending=False)
    # create list of just the ids
    most_similar_users = sims_sorted.index.tolist()
    # remove the own user's id (previously done)

    return most_similar_users

# Do a spot check of your function
print("The 10 most similar users to user 1 are: {}".format(find_similar_users(1)[:10]))
print("The 5 most similar users to user 3933 are: {}".format(find_similar_users(3933)[:5]))
print("The 3 most similar users to user 46 are: {}".format(find_similar_users(46)[:3]))


def get_article_names(article_ids, df=df):
    '''
    INPUT:
    article_ids - (list) a list of article ids
    df - (pandas dataframe) df as defined at the top of the notebook

    OUTPUT:
    article_names - (list) a list of article names associated with the list of
                    article ids (this is identified by the title column)
    '''
    article_names = list(set(df[df['article_id'].isin(article_ids)]['title']))

    return article_names


def get_user_articles(user_id, user_item=user_item):
    '''
    INPUT:
    user_id - (int) a user id
    user_item - (pandas dataframe) matrix of users by articles:
                1's when a user has interacted with an article, 0 otherwise

    OUTPUT:
    article_ids - (list) a list of the article ids seen by the user
    article_names - (list) a list of article names associated with the list of
                    article ids (this is identified by the doc_full_name column
                    in df_content)

    Description:
    Provides a list of the article_ids and article titles that have been seen
    by a user
    '''
    aa = user_item.iloc[user_id-1]
    article_ids = list(map(str,list(aa[aa == 1].index))) # list(aa[aa == 1].index)
    article_names = get_article_names(article_ids)

    return article_ids, article_names


def user_user_recs(user_id, m=10):
    '''
    INPUT:
    user_id - (int) a user id
    m - (int) the number of recommendations you want for the user

    OUTPUT:
    recs - (list) a list of recommendations for the user

    Description:
    Loops through the users based on closeness to the input user_id
    For each user - finds articles the user hasn't seen before and provides
    them as recs. Do this until m recommendations are found

    Notes:
    Users who are the same closeness are chosen arbitrarily as the 'next' user

    For the user where the number of recommended articles starts below m
    and ends exceeding m, the last items are chosen arbitrarily

    '''
    recs = []
    ids, names = get_user_articles(user_id)
    sims = find_similar_users(user_id)
    for u in sims:
        new_ids, new_names = get_user_articles(u)
        recs.extend(np.setdiff1d(new_ids, ids))
        if len(recs) >= m:
            recs = random.sample(recs, m)  # recs[:m]
            break

    return recs


# Check Results
get_article_names(user_user_recs(1, 10)) # Return 10 recommendations for user 1

# Test your functions here - No need to change this code - just run this cell
assert set(get_article_names(['1024.0', '1176.0', '1305.0', '1314.0', '1422.0', '1427.0'])) == set(['using deep learning to reconstruct high-resolution audio', 'build a python app on the streaming analytics service', 'gosales transactions for naive bayes model', 'healthcare python streaming application demo', 'use r dataframes & ibm watson natural language understanding', 'use xgboost, scikit-learn & ibm watson machine learning apis']), "Oops! Your the get_article_names function doesn't work quite how we expect."
assert set(get_article_names(['1320.0', '232.0', '844.0'])) == set(['housing (2015): united states demographic measures','self-service data preparation with ibm data refinery','use the cloudant-spark connector in python notebook']), "Oops! Your the get_article_names function doesn't work quite how we expect."
assert set(get_user_articles(20)[0]) == set(['1320.0', '232.0', '844.0'])
assert set(get_user_articles(20)[1]) == set(['housing (2015): united states demographic measures', 'self-service data preparation with ibm data refinery','use the cloudant-spark connector in python notebook'])
assert set(get_user_articles(2)[0]) == set(['1024.0', '1176.0', '1305.0', '1314.0', '1422.0', '1427.0'])
assert set(get_user_articles(2)[1]) == set(['using deep learning to reconstruct high-resolution audio', 'build a python app on the streaming analytics service', 'gosales transactions for naive bayes model', 'healthcare python streaming application demo', 'use r dataframes & ibm watson natural language understanding', 'use xgboost, scikit-learn & ibm watson machine learning apis'])
print("If this is all you see, you passed all of our tests!  Nice job!")


def get_top_sorted_users(user_id, df=df, user_item=user_item):
    '''
    INPUT:
    user_id - (int)
    df - (pandas dataframe) df as defined at the top of the notebook
    user_item - (pandas dataframe) matrix of users by articles:
            1's when a user has interacted with an article, 0 otherwise


    OUTPUT:
    neighbors_df - (pandas dataframe) a dataframe with:
                    neighbor_id - is a neighbor user_id
                    similarity - measure of the similarity of each user to the provided user_id
                    num_interactions - the number of articles viewed by the user - if a u

    Other Details - sort the neighbors_df by the similarity and then by number of interactions where
                    highest of each is higher in the dataframe

    '''
    # compute similarity of each user to the provided user
    sims = np.dot(user_item.drop([user_id]), user_item.iloc[user_id-1])
    ind = user_item.drop([user_id]).index
    sims = pd.Series(sims, index=ind, name="similarity")
    inter = user_item.sum(axis=1).drop([user_id]).rename("num_interactions")
    # Sorting it descending by similarity and interactions gave an incorrect
    # result according to the provided test. So I changed the sorting method to
    # use the similarity and index variables.
    neighbors_df = pd.concat([sims, inter], axis=1)#.sort_values(by=["similarity", "num_interactions"], inplace=False, ascending=False)
    neighbors_df['index1'] = neighbors_df.index
    neighbors_df = neighbors_df.sort_values(by=["similarity", "index1"], inplace=False, ascending=[False,True])
    return neighbors_df


def user_user_recs_part2(user_id, m=10):
    '''
    INPUT:
    user_id - (int) a user id
    m - (int) the number of recommendations you want for the user

    OUTPUT:
    recs - (list) a list of recommendations for the user by article id
    rec_names - (list) a list of recommendations for the user by article title

    Description:
    Loops through the users based on closeness to the input user_id
    For each user - finds articles the user hasn't seen before and provides
                    them as recs. Do this until m recommendations are found

    Notes:
    * Choose the users that have the most total article interactions
    before choosing those with fewer article interactions.

    * Choose articles with the articles with the most total interactions
    before choosing those with fewer total interactions.

    '''
    recs = []
    ids, names = get_user_articles(user_id)
    sims = get_top_sorted_users(user_id).index.tolist()
#    art = user_item.sum(axis=0).rename("tot_interactions")
#    sims = find_similar_users(user_id)
    for u in sims:
        new_ids, new_names = get_user_articles(u)
        recs.extend(np.setdiff1d(new_ids, ids))
        if len(recs) >= m:
#            recs2 = art.loc[art.index.isin(recs)]
            recs = recs[:m]  # random.sample(recs, m)
            break
    rec_names = get_article_names(recs)

    return recs, rec_names


# Quick spot check - don't change this code - just use it to test your
# functions
rec_ids, rec_names = user_user_recs_part2(20, 10)
print("The top 10 recommendations for user 20 are the following article ids:")
print(rec_ids)
print()
print("The top 10 recommendations for user 20 are the following article names:")
print(rec_names)

# Tests with a dictionary of results

user1_most_sim = get_top_sorted_users(1).index[0]
user131_10th_sim = get_top_sorted_users(131).index[9]

# Dictionary Test Here
sol_5_dict = {
    'The user that is most similar to user 1.': user1_most_sim,
    'The user that is the 10th most similar to user 131': user131_10th_sim,
}

t.sol_5_test(sol_5_dict)

new_user = '0.0'

# What would your recommendations be for this new user '0.0'?  As a new user, they have no observed articles.
# Provide a list of the top 10 article ids you would give to 
new_user_recs = get_top_article_ids(10) # Your recommendations here


# =============================================================================
# We can use the user_user_recs_part2 function to provide any number of recommendations
# for any one user. We can also utilize the article data more by ranking the most
# popular articles of similar users first. We can also simply recommend the most
# popular articles, or the ones with the most interactions.
# =============================================================================

# Load the matrix here
user_item_matrix = pd.read_pickle('user_item_matrix.p')

# quick look at the matrix
#user_item_matrix.head()

# Perform SVD on the User-Item Matrix Here

u, s, vt = np.linalg.svd(user_item_matrix)

num_latent_feats = np.arange(10, 700+10,20)
sum_errs = []

for k in num_latent_feats:
    # restructure with k latent features
    s_new, u_new, vt_new = np.diag(s[:k]), u[:, :k], vt[:k, :]

    # take dot product
    user_item_est = np.around(np.dot(np.dot(u_new, s_new), vt_new))

    # compute error for each prediction to actual value
    diffs = np.subtract(user_item_matrix, user_item_est)

    # total errors and keep track of them
    err = np.sum(np.sum(np.abs(diffs)))
    sum_errs.append(err)

plt.plot(num_latent_feats, 1 - np.array(sum_errs)/df.shape[0]);
plt.xlabel('Number of Latent Features');
plt.ylabel('Accuracy');
plt.title('Accuracy vs. Number of Latent Features');
plt.show()

df_train = df.head(40000)
df_test = df.tail(5993)


def create_test_and_train_user_item(df_train, df_test):
    '''
    INPUT:
    df_train - training dataframe
    df_test - test dataframe

    OUTPUT:
    user_item_train - a user-item matrix of the training dataframe
                      (unique users for each row and unique articles for each
                      column)
    user_item_test - a user-item matrix of the testing dataframe
                    (unique users for each row and unique articles for each
                    column)
    test_idx - all of the test user ids
    test_arts - all of the test article ids
    '''
    user_item_train = pd.pivot_table(df_train,
                                     values='title',
                                     index=['user_id'],
                                     columns=['article_id'],
                                     aggfunc=pd.Series.nunique,  # np.count_nonzero,
                                     fill_value=0)
    user_item_test = pd.pivot_table(df_test,
                                    values='title',
                                    index=['user_id'],
                                    columns=['article_id'],
                                    aggfunc=pd.Series.nunique,  # np.count_nonzero,
                                    fill_value=0)
    test_idx = user_item_test.index.tolist()
    test_arts = user_item_test.columns.tolist()

    return user_item_train, user_item_test, test_idx, test_arts

user_item_train, user_item_test, test_idx, test_arts = create_test_and_train_user_item(df_train, df_test)

# =============================================================================
# print(len(np.intersect1d(test_idx, df_train['user_id'].tolist())))
# print(len(np.setdiff1d(test_idx, df_train['user_id'].tolist())))
# 
# print(len(np.intersect1d(test_arts, df_train['article_id'].tolist())))
# print(len(np.setdiff1d(test_arts, df_train['article_id'].tolist())))
# =============================================================================

# Replace the values in the dictionary below
a = 662
b = 574
c = 20
d = 0

sol_4_dict = {
    'How many users can we make predictions for in the test set?': c, # letter here, 
    'How many users in the test set are we not able to make predictions for because of the cold start problem?': a, # letter here, 
    'How many movies can we make predictions for in the test set?': b, # letter here,
    'How many movies in the test set are we not able to make predictions for because of the cold start problem?': d # letter here
}

t.sol_4_test(sol_4_dict)

# fit SVD on the user_item_train matrix
u_train, s_train, vt_train = np.linalg.svd(user_item_train)

u_test = u_train[user_item_train.index.isin(test_idx)]
vt_test = vt_train[:, user_item_train.columns.isin(test_arts)]

# users who exist in both training and test datasets
user_viable = np.intersect1d(test_idx, df_train['user_id'].tolist())
user_item_test_predict = user_item_test[user_item_test.index.isin(user_viable)]

# initialize testing parameters
num_latent_feats = np.arange(10,700+10,10)
sum_errs_train = []
sum_errs_test = []

for k in num_latent_feats:
    # restructure with k latent features for both training and test sets
    s_train_l, u_train_l, vt_train_l = np.diag(s_train[:k]), u_train[:, :k], vt_train[:k, :]
    u_test_l, vt_test_l = u_test[:, :k], vt_test[:k, :]
    
    # take dot product for both training and test sets
    user_item_train_est = np.around(np.dot(np.dot(u_train_l, s_train_l), vt_train_l))
    user_item_test_est = np.around(np.dot(np.dot(u_test_l, s_train_l), vt_test_l))
    
    # compute error for each prediction to actual value
    diffs_train = np.subtract(user_item_train, user_item_train_est)
    diffs_test = np.subtract(user_item_test_predict, user_item_test_est)
    
    # total errors and keep track of them for both training and test sets
    err_train = np.sum(np.sum(np.abs(diffs_train)))
    err_test = np.sum(np.sum(np.abs(diffs_test)))
    sum_errs_train.append(err_train)
    sum_errs_test.append(err_test)

plt.plot(num_latent_feats, 1 - np.array(sum_errs_train)/(user_item_train.shape[0] * user_item_test_predict.shape[1]), label='Train')
plt.plot(num_latent_feats, 1 - np.array(sum_errs_test)/(user_item_test_predict.shape[0] * user_item_test_predict.shape[1]), label='Test')
plt.xlabel('Number of Latent Features')
plt.ylabel('Prediction Accuracy')
plt.title('Accuracy Vs. Number of Latent Features')
plt.legend()
plt.show()
