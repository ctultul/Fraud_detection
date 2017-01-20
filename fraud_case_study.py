from __future__ import division
import cPickle as pickle
import pandas as pd
import numpy as np
import unicodedata
import matplotlib.pyplot as plt
import datetime
from bs4 import BeautifulSoup
from time import gmtime
import string
from sklearn import tree
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import precision_score, recall_score, roc_curve, confusion_matrix
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def dummify(df,column):
    dummy = pd.get_dummies(df[column]).rename(columns=lambda x: column + ' ' + str(x)).iloc[:,0:len(df[column].unique())]

    df.drop(column, axis=1, inplace=True)
    return pd.concat([df,dummy], axis=1)

def dummify_all_nonnumeric_cols(df):
    for col in df.columns:
        if not isinstance(df[col][0],float) and not isinstance(df[col][0],int):
            df = dummify(df,col)
    return df

def convert_nans_to_means(df):
    for col in df.columns:
        if isinstance(df[col][0],int) or isinstance(df[col][0],float):
            df[col].fillna(df[col].mean())
    return df

def add_comment_text_info(df):
    num_exclaimation = []
    num_question = []
    num_words = []
    num_upper_words = []
    count = lambda l1,l2: sum([1 for x in l1 if x in l2])
    for desc in df['description']:
        upper = 0
        num_exclaimation.append(count(desc, string.punctuation[0]))
        num_question.append(count(desc, string.punctuation[20]))
        num_words.append(count(desc, " "))
        for word in desc.split(" "):
            if word.isupper():
                upper +=1
        num_upper_words.append(upper)
    df['num_exclaimation'] = num_exclaimation
    df['num_question'] = num_question
    df['num_words'] = num_words
    df['num_upper_words'] = num_upper_words

    return df

def add_descrip_only_text_column(df):
    des_text = []
    for des in df['description']:
        des_text.append(BeautifulSoup(des).text)
    df['des_text'] = des_text
    return df

def edit_acct_type_col(df):
    fraud_terms= ['fraudster_event', u'fraudster', u'fraudster_att']
    spammer_terms= ['spammer_warn','spammer_limited', 'spammer_noinvite','spammer_web']

    df['acct_type'] = df['acct_type'].replace(spammer_terms,'spammer')
    df['acct_type'] = df['acct_type'].replace(fraud_terms,'fraudster')

    return df

def filter_data(df):
    # df['description'] = df.description.apply(lambda des: unicodedata.normalize('NFKD', des).encode('ascii','ignore'))
    # df['approx_payout_date'] = df.approx_payout_date.apply(gmtime)

    df = df.reset_index(drop= True)
    df = edit_acct_type_col(df)
    df = add_comment_text_info(df)
    df.user_type[df.user_type > 4] = 4 # should be in range 1-4, if higher, error
    df= convert_nans_to_means(df)

    df = df[['fraud','user_age','user_type','body_length','num_question', 'num_words', 'num_upper_words','num_exclaimation']]

    return df

def get_data(path):
    df = pd.read_json(path)
    df = edit_acct_type_col(df)

    df['fraud'] = [True if acct == 'fraudster' else False for acct in df['acct_type']]

    y = df['fraud']
    X = df.drop(['fraud'], axis=1)
    return X, y

def train_test_split_func(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4, stratify=X['acct_type'])
    return X_train, X_test, y_train, y_test

    # print "LR_Accuracy:", accuracy
    # print "LR_Precision:", precision_score(y_test, y_predicted)
    # print "LR_Recall:", recall_score(y_test, y_predicted)

def model_for_testing_dp(df, train=False):
    df = edit_acct_type_col(df)

    df['fraud'] = [True if acct == 'fraudster' else False for acct in df['acct_type']]
    df = filter_data(df)

    y = df['fraud']
    X = df.drop(['fraud'], axis=1)

    X = dummify_all_nonnumeric_cols(X)

    return X, y

def oversample(X, y, target):
    '''
    Input: X, y data frames. Target = target increase of minority data.

    Output: X, y pandas dataframes/series of oversampled data + old data.
    '''
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    positive_count = sum(y)
    negative_count = len(y) - positive_count
    target_positive_count = target*negative_count / (1. - target)
    target_positive_count = int(round(target_positive_count))
    number_of_new_observations = target_positive_count - positive_count
    positive_obs_indices = np.where(y==True)[0]
    new_obs_indices = np.random.choice(positive_obs_indices,
                                       size=number_of_new_observations,
                                       replace=True)

    X_new, y_new = X.iloc[new_obs_indices], y.iloc[new_obs_indices]
    X_positive = pd.concat((X.iloc[positive_obs_indices], X_new))
    y_positive = pd.concat([y.iloc[positive_obs_indices], y_new])
    X_negative = X[y==False]
    y_negative = y[y==False]

    X_oversampled = pd.concat([X_negative, X_positive])
    y_oversampled = pd.concat([y_negative, y_positive])
    return X_oversampled, y_oversampled

def GB_model(X, y):
    '''
    GradientBoostingClassifier
    Input: pandas dataframes

    Output: accuracy, precision, recall, y_pred, y_probas
    '''
    GB = GradientBoostingClassifier()
    GB.fit(X,y)
    return GB

def predict_off_GB(fitted_model, X_test, y_test):
    y_pred = fitted_model.predict(X_test)
    accuracy = fitted_model.score(X_test, y_test)

    print 'GB: {}'.format(accuracy)

def RF_model(X, y):
    '''
    RandomForestClassifier
    Input: pandas dataframes

    Output: accuracy, precision, recall, y_pred, y_probas
    '''
    RF = RandomForestClassifier()
    RF.fit(X, y)
    y_pred = RF.predict(X)
    y_probas = RF.predict_proba(X)[:,1]

    accuracy = cross_val_score(RF, X, y).mean()
    precision = cross_val_score(RF, X, y, scoring='precision').mean()
    recall = cross_val_score(RF, X, y, scoring='recall').mean()


    # y_pred_test = RF.predict(X_test)
    # y_probas_test = RF.predict_proba(X_test)[:,1]

    # feature_importances = np.argsort(RF.feature_importances_)
    # values_for_graphing = RF.feature_importances_[feature_importances[-1:-11:-1]]
    #
    # importances = list(X_train.columns[feature_importances[-1:-11:-1]])
    # print "Top ten features:", importances
    #
    # std = np.std([tree.feature_importances_ for tree in RF.estimators_], axis=0)
    # indices = np.argsort(importances)[::-1]

    # plt.figure()
    # plt.title("Feature importances")
    # plt.barh(len(values_for_graphing), values_for_graphing, color="g", align="center")
    # plt.xlabel('Score')
    # plt.ylim([-1, 10])
    # plt.show()

    # fpr_RF, tpr_RF, thresholds = roc_curve(y_train, y_probas)
    # plt.plot(fpr_RF, tpr_RF, label='Random Forest')
    # plt.xlabel("False Positive Rate (1 - Specificity)")
    # plt.ylabel("True Positive Rate (Sensitivity, Recall)")
    # # plt.title("ROC plot for Random Forest")
    # plt.show()

    print 'RF: {},{},{}'.format(accuracy,precision,recall)

    # return accuracy, precision, recall, y_pred, y_probas, fpr_RF, tpr_RF

def LR_model(X_train, y_train):
    LR = LogisticRegression()
    accuracy = cross_val_score(LR, X_train, y_train).mean()
    LR.fit(X_train, y_train)
    precision = cross_val_score(LR, X_train, y_train, scoring='precision').mean()
    recall = cross_val_score(LR, X_train, y_train, scoring='recall').mean()
    return accuracy, precision, recall

if __name__ == '__main__':
    X, y = get_data('data/data.json')
    X_train, X_test, y_train, y_test = train_test_split_func(X, y)

    X_train_os, y_train_os = oversample(X_train, y_train, 0.3)

    df_train = pd.concat([pd.DataFrame(y_train_os),X_train_os], axis=1)

    df_test = pd.concat([pd.DataFrame(y_test),X_test], axis=1)

    X_train_gb, y_train_gb = model_for_testing_dp(df_train,train=True)

    X_test_gb, y_test_gb  = model_for_testing_dp(df_test)

    #columns_to_use = [col for col in columns_tr if col in columns_tst]

    #X_test_gb = X_test_gb[columns_to_use]
    #y_test_gb = y_test_gb[columns_to_use]

    # predict_off_GB(fitted_model, X_test_gb, y_test_gb)

    model = GradientBoostingClassifier()
    model.fit(X_train_gb, y_train_gb)

    with open('data/model.pkl', 'w') as f:
        pickle.dump(model, f)

    # with open('data/model.pkl', 'w') as f:
    #     fitted_model = pickle.load(f)
    #
    # fitted_model.predict(X_test_gb)



    # y_limited_train = df_train['fraud']
    # X_limited_train = df_train.drop(['fraud'], axis=1)
    #
    # accuracy, precision, recall, y_pred, y_probas, fpr_RF, tpr_RF = RF_model(X_limited_train, y_limited_train)
    # accuracy, precision, recall = LR_model(X_limited_train, y_limited_train)
