from __future__ import division
import pandas as pd
import numpy as np
import unicodedata
import matplotlib.pyplot as plt
import datetime
from bs4 import BeautifulSoup
from time import gmtime
from sklearn import tree
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import precision_score, recall_score, roc_curve, confusion_matrix
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def dummify(df,column):
    '''
    Input: pandas dataframe
    Output: pandas dataframe

    This function dummifies any categorical variable that you pass through it.
    '''
    dummy = pd.get_dummies(df[column]).rename(columns=lambda x: column+'_'+str(x)).iloc[:,0:len(df[column].unique())]
    df = df.drop(column,axis=1)
    return pd.concat([df,dummy],axis=1)




def add_descrip_only_text_column(df):
    '''
    Input: dataframe
    Output: dataframe

    This function adds a text column to our dataframe, which is comprised of text from the description column in our dataframe.
    '''
    des_text = []
    for des in df['description']:
        des_text.append(BeautifulSoup(des, "lxml").text)
    df['des_text'] = des_text
    return df




def convert_nans_to_means(df):
    '''
    Input: pandas df
    Output: pandas df

    If elements of a column are integers or floats, this function converts nans to mode. May change to mean if desired.
    '''
    for col in df.columns:
        if isinstance(df[col].iloc[0],int) or isinstance(df[col].iloc[0],float):
            df[col].fillna(df[col].mode())
    return df




def get_data(dataframe):
    '''
    Input: pandas dataframe

    Output: y, X dataframes

    Manipulates/cleans the data
    '''
    df = pd.read_json(dataframe)
    df = convert_nans_to_means(df)

    df = dummify(df,'country')
    df['constant'] = 1

    df['description'] = [unicodedata.normalize('NFKD', des).encode('ascii','ignore') for des in df['description']]

    df['approx_payout_date'] = df.approx_payout_date.apply(gmtime)

    fraud_terms = ['fraudster_event', u'fraudster', u'fraudster_att']
    spammer_terms = ['spammer_warn','spammer_limited', 'spammer_noinvite','spammer_web']

    df['acct_type'] = df['acct_type'].replace(spammer_terms,'spammer')

    df['acct_type'] = df['acct_type'].replace(fraud_terms,'fraudster')

    df['fraud'] = [True if event in fraud_terms else False for event in df['acct_type']]

    df.user_type[df.user_type > 4] = 4 # should be in range 1-4, if higher, error

    df = add_descrip_only_text_column(df)

    y = df['fraud']
    X = df.drop(['fraud'], axis=1)
    return y, X




def train_test_split_func(X, y):
    '''
    Splits our data into a training set and a testing set. Test size is set to 0.2.
    Using random_state=4
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4, stratify=X['acct_type'])
    return X_train, X_test, y_train, y_test




def LR_model(X_train, y_train, X_test, y_test):
    '''
    LogisticRegression Model
    Input: pandas dataframes

    Output: confusion_matrix, accuracy, precision, recall, y_pred, y_probas
    '''
    LR = LogisticRegression()
    LR.fit(X_train, y_train)
    y_probas = LR.predict_proba(X_train)[:,1]
    y_pred = LR.predict(X_train)

    cm = confusion_matrix(y_pred, y_train)
    accuracy = cross_val_score(LR, X_train, y_train).mean()
    precision = cross_val_score(LR, X_train, y=y_train, scoring='precision').mean()
    recall = cross_val_score(LR, X_train, y_train, scoring='recall').mean()

    # y_predicted = LR.predict(X_test)
    # y_predicted_probas = LR.predict_proba(X_test)[:,1]

    return cm, accuracy, precision, recall, y_pred, y_probas




def roc_curve_logit(probabilities, labels):
    '''
    ROC curve function.
    Input: probabilities and labels

    Output: true positive rates, false positive rates and thresholds
    '''
    thresholds = np.sort(probabilities)

    tprs = []
    fprs = []

    num_positive_cases = sum(labels)
    num_negative_cases = len(labels) - num_positive_cases
    for threshold in thresholds:
       predicted_positive = probabilities >= threshold
       true_positives = np.sum(predicted_positive * labels)
       false_positives = np.sum(predicted_positive) - true_positives
       tpr = true_positives / float(num_positive_cases)
       fpr = false_positives / float(num_negative_cases)

       fprs.append(fpr)
       tprs.append(tpr)

    return tprs, fprs, thresholds.tolist()




def DT_model(X_train, y_train, X_test, y_test):
    '''
    DecisionTreeClassifier
    Input: pandas dataframes

    Output: accuracy, precision, recall, y_pred, y_probas
    '''
    DT = tree.DecisionTreeClassifier()
    DT.fit(X_train, y_train)
    y_pred = DT.predict(X_train)
    y_probas = DT.predict_proba(X_train)[:,1]

    accuracy = cross_val_score(DT, X_train, y_train).mean()
    precision = cross_val_score(DT, X_train, y=y_train, scoring='precision').mean()
    recall = cross_val_score(DT, X_train, y_train, scoring='recall').mean()
    # y_predicted = DT.predict(X_test)
    # y_predicted_probas = DT.predict_proba(X_test)[:,1]

    return accuracy, precision, recall, y_pred, y_probas




def RF_model(X_train, y_train, X_test, y_test):
    '''
    RandomForestClassifier
    Input: pandas dataframes

    Output: accuracy, precision, recall, y_pred, y_probas
    '''
    RF = RandomForestClassifier()
    accuracy = cross_val_score(RF, X_train, y_train).mean()
    RF.fit(X_train, y_train)
    y_pred = RF.predict(X_train)
    y_probas = RF.predict_proba(X_train)[:,1]
    precision = cross_val_score(RF, X_train, y=y_train, scoring='precision').mean()
    recall = cross_val_score(RF, X_train, y_train, scoring='recall').mean()

    # y_pred_test = RF.predict(X_test)
    # y_probas_test = RF.predict_proba(X_test)[:,1]


    feature_importances = np.argsort(RF.feature_importances_)
    values_for_graphing = RF.feature_importances_[feature_importances[-1:-11:-1]]

    importances = list(X_train.columns[feature_importances[-1:-11:-1]])
    print "Top ten features:", importances

    std = np.std([tree.feature_importances_ for tree in RF.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

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

    return accuracy, precision, recall, y_pred, y_probas, fpr_RF, tpr_RF




def GBC(X_train, y_train):
    '''
    GradientBoostingClassifier
    Input: X, y data as pandas dataframes.

    Output: grid search of best score and best params

    Not complete!
    '''
    est = GradientBoostingClassifier()
    param_grid = {'learning_rate': [0.1, 0.05, 0.02],
                 'max_depth': [2, 3],
                 'min_samples_leaf': [3, 5],}

    gs_cv = GridSearchCV(est, param_grid, n_jobs=2).fit(X_train, y_train)

    return gs_cv.best_score_, gs_cv.best_params_




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




if __name__ == '__main__':
    '''
    get_data is used to TTS. Test data is not used. Train data is used for cross validation in our models.
    '''
    y, X = get_data('data/data.json')
    X_train, X_test, y_train, y_test = train_test_split_func(X, y)

    # Oversample minority data (target=0.3)
    X_train_os, y_train_os = oversample(X_train, y_train, 0.3)


    # Limited train data used just to run the models initially
    X_limited_train = X_train_os[['body_length', 'user_age', 'country_MA','user_type', 'constant']]
    y_limited_train = y_train_os.astype(int)


    # More features included -- as of now used for random forest to determine important features
    X_limited_train_1 = X_train_os.drop(['acct_type', 'description', 'des_text'], axis=1)



    '''
    Get cm, accuracy, precision, recall, predictions and prediction probabilities for 3 models: LR, DT, RF

    For RandomForest, used all variables (no feature engineering), except account type (acct_type will cause leakage.
    '''
    cm, accuracy_LR, precision_LR, recall_LR, y_pred_LR, y_probas_LR = LR_model(X_limited_train, y_limited_train, X_test, y_test)

    accuracy_DT, precision_DT, recall_DT, y_pred_DT, y_probas_DT = DT_model(X_limited_train, y_limited_train, X_test, y_test)

    accuracy_RF, precision_RF, recall_RF, y_pred_RF, y_probas_RF, fprs_RF, tprs_RF = RF_model(X_limited_train_1, y_limited_train, X_test, y_test)


    '''
    Plot ROC curves for LR and DT
    '''
    tprs_LR, fprs_LR, thresholds_LR = roc_curve_logit(y_probas_LR, y_limited_train)

    tprs_DT, fprs_DT, thresholds_DT = roc_curve_logit(y_probas_DT, y_limited_train)


    plt.plot(fprs_LR, tprs_LR, label='LogisticR')
    plt.plot(fprs_DT, tprs_DT, label='DecisionTrees')
    # plt.plot(fpr_RF, tpr_RF, label='RF')
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity, Recall)")
    plt.title('ROC Curve')
    # plt.xlim((0.7,1.0))
    # plt.ylim((0.95,1.0))
    plt.legend(loc=4)
    plt.show()



    '''
    Working with test data
    '''
    # X_limited_test = X_test[['body_length', 'delivery_method', 'user_age', 'user_type']]
