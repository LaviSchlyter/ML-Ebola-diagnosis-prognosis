# -*- coding: utf-8 -*-
"""Some helper functions for project 2."""
###### Basics

import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import graphviz
import statsmodels.api as sm

##### Sklearn
## metrics

from sklearn.metrics import f1_score,accuracy_score,precision_recall_curve,roc_auc_score,roc_curve, auc

## models

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LassoCV
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from sklearn import tree
from sklearn.feature_selection import RFE

#### yellowbrick

from yellowbrick.classifier.rocauc import roc_auc
from yellowbrick.classifier import ConfusionMatrix, ClassificationReport, ClassPredictionError, ROCAUC,PrecisionRecallCurve
from yellowbrick.features import PCA as PCA_3D
from yellowbrick.features import Rank2D
from yellowbrick.target import ClassBalance
from yellowbrick.features.radviz import radviz

########################################### FEATURE - ENGINEERING ############################################################

def delete_minus1(X): 
    """ Delete columns and rows which contain unknowns (-1)
    we first start by removing columns and then the rows

    Parameters
    ----------
    X: matrix of features

    Returns
    -------
    
    - Xp : A matrix with no missing values
    
    """
    Xp = X.copy()
    for column in Xp:

        if sum(Xp[column]==-1) >= 0.5 * len(Xp[column]):
            Xp.drop(column, axis=1, inplace = True)
            continue

        index_names = Xp[(Xp[column] == -1)].index;
        Xp.drop(index_names, inplace = True)
    
    return Xp



def make_ones_indicator_column(data_frame, name_of_column_target, inplace=False):
    """ For each column we add an indicator column which adds a '1' for when the information 
    is known (1 or 0) and adds a '0' for the data is unknown (-1). Same name as the target column and suffix _indicator
    If the indicator column already exists: function does nothing.
    
    Parameters
    ----------
    data_frame: A data frame
    
    name_of_column_target: Column which will be given an indicator column
    
    inplace = State whether the cahnge must be a copy or an inplace operation (Default = False)

    Returns
    -------
    
    - Returns the indicator column filled with ones and zeros
    
    """
    if name_of_column_target + '_indicator' in data_frame.columns:
        if not inplace:
            return data_frame.copy()
    else:
        if inplace:
            data_frame[name_of_column_target + '_indicator'] = np.ones(data_frame[name_of_column_target].size)
        else :
            df_temp = data_frame.copy()
            df_temp[name_of_column_target + '_indicator'] = np.ones(df_temp[name_of_column_target].size)
            return df_temp



def put_zero_in_indicator_column(data_frame, name_of_column_target, target_value, inplace=False):
    """ Finds in the indicator column the lines where the target column has target value, and puts 0 there
    

    Parameters
    ----------
    data_frame: A data frame
    
    name_of_column_target: Column which will be given an indicator column
    
    target_value: value that will be change to zero each time it is encountered

    inplace = State whether the cahnge must be a copy or an inplace operation (Default = False)
    Returns
    -------
    
    - A data frame where zeros will be added to the indicator column
    
    """
    if inplace:
        data_frame.loc[data_frame[name_of_column_target] == target_value, name_of_column_target + '_indicator'] = 0
    else :
        df_temp = data_frame.copy()
        df_temp.loc[df_temp[name_of_column_target] == target_value, name_of_column_target + '_indicator'] = 0
        return df_temp
    



# adds a column to the right, with name of target + _indicator, with 0 on same line as the target column has target value

def make_indicator_for_bad_data(data_frame, name_of_column_target, target_value, inplace=False):
    """ Looking at 

    Parameters
    ----------
    X: matrix of features

    Returns
    -------
    
    - A plot with correlation features
    
    """
    if inplace:
        make_ones_indicator_column(data_frame, name_of_column_target, inplace)
        put_zero_in_indicator_column(data_frame, name_of_column_target, target_value, inplace)
    else :
        df_temp = make_ones_indicator_column(data_frame, name_of_column_target, inplace=False)
        put_zero_in_indicator_column(df_temp, name_of_column_target, target_value, inplace=True)
        return df_temp



# Makes columns iteratively using make_indicator_for_bad_data
# putting several times the same column name in list to search for multiple targets in that column.

def make_indicators(data_frame, list_of_column_target, list_of_target_values, inplace=False):
    """ Looking at 

    Parameters
    ----------
    X: matrix of features

    Returns
    -------
    
    - A plot with correlation features
    
    """
    if not inplace:
        df_temp = data_frame.copy()
    else:
        df_temp = data_frame
    for i, col in enumerate(list_of_column_target):
        make_indicator_for_bad_data(df_temp, col, list_of_target_values[i], inplace=True)
    if not inplace:
        return df_temp
    



from sklearn.impute import SimpleImputer

def handle_missing_values(X, target_values):
    """ Using KNN algorithm to predict missing values

    Parameters
    ----------
    X: matrix of features

    Returns
    -------
    
    - A plot with correlation features
    
    """
    
    imputer = SimpleImputer(missing_values= target_values)
    mod = imputer.fit(X)
    X_trans = pd.DataFrame(mod.transform(X))
    X_trans.columns = X.columns
    return X_trans

 

# to extract just one subset of data:
def extract_certain_dataset(data_frame, name_of_target_column, target_value):
    """ Using KNN algorithm to predict missing values

    Parameters
    ----------
    X: matrix of features

    Returns
    -------
    
    - A plot with correlation features
    
    """
    df = data_frame[data_frame[name_of_target_column] == target_value].copy()
    return df


# to separate the dataframe in subgroups depending on the values in the target column.
def make_list_by_value(data_frame, name_of_target_column, name_of_reference_column):
    """ Using KNN algorithm to predict missing values

    Parameters
    ----------
    X: matrix of features

    Returns
    -------
    
    - A plot with correlation features
    
    """
    list_of_df = []
    list_of_target_values = data_frame[name_of_target_column].value_counts().index.sort_values()
    for i, value in enumerate(list_of_target_values):
        list_of_df.append(extract_certain_dataset(data_frame, name_of_target_column, value).set_index(name_of_reference_column))
    return list_of_df


# first separates the original dataset into subgroups corresponding to values on the target column,
# and then puts them back toghether, but horizontally intead of vertically. Aloigned around the reference column.
def rearange_horizontally(data_frame, name_of_target_column, name_of_reference_column):
    """ Using KNN algorithm to predict missing values

    Parameters
    ----------
    X: matrix of features

    Returns
    -------
    
    - A plot with correlation features
    
    """
    list_of_df = make_list_by_value(data_frame, name_of_target_column, name_of_reference_column)
    return pd.concat(list_of_df, axis=1, sort=False)#.reset_index()
    

# ouputs a list of sub-dataframes. One for each different value in the targeted column 
# (slightly different from make_list_by_value because of context of use)
def dissassemble(data_frame, name_of_column_target):
    """ Using KNN algorithm to predict missing values

    Parameters
    ----------
    X: matrix of features

    Returns
    -------
    
    - A plot with correlation features
    
    """
    out_list=[]
    for i, value in enumerate(data_frame[name_of_column_target].value_counts(sort=False).index):
        df = data_frame[data_frame[name_of_column_target] == value].copy()
        out_list.append(df)
    return out_list


# outputs a list of lists of sub-dataframes. (calls dissassemble twice)
def fine_dissassembly(data_frame, name_first_column_target, name_second_column_target):
    """ Using KNN algorithm to predict missing values

    Parameters
    ----------
    X: matrix of features

    Returns
    -------
    
    - A plot with correlation features
    
    """
    out_list = dissassemble(data_frame, name_first_column_target)
    for i, subdf in enumerate(out_list):
        out_list[i] = dissassemble(subdf, name_second_column_target)
    return out_list


# Undoes dissassemble
def reassemble(list_of_df):
    """ Using KNN algorithm to predict missing values

    Parameters
    ----------
    X: matrix of features

    Returns
    -------
    
    - A plot with correlation features
    
    """
    return pd.concat(list_of_df)


# Undoes fine_dissassembly
def big_reassembly(list_of_list_of_df):
    """ Using KNN algorithm to predict missing values

    Parameters
    ----------
    X: matrix of features

    Returns
    -------
    
    - A plot with correlation features
    
    """
    for i, sublist in enumerate(list_of_list_of_df):
        list_of_list_of_df[i] = reassemble(sublist)
    return reassemble(list_of_list_of_df)
   

def subtract_list(list1, list2):
    """ Using KNN algorithm to predict missing values

    Parameters
    ----------
    X: matrix of features

    Returns
    -------
    
    - A plot with correlation features
    
    """
    list3= []
    for i in list1:
        if i not in list2:
            list3.append(i)
    return list3


# just running df = transform_into_horizontal_df(data_frame) will do all the work, provided the dataset does not have more than one observation per day.
# Will separate the dataset into smaller datasets corresponding to the values of 'time_elapsed' and then concatenate everything horizontally.
def transform_into_horizontal_df(data_frame, 
                                 reference_column='msfid', 
                                 current_time_column='datclin', 
                                 first_day_column='first_date', 
                                 no_need_duplicate_columns=['sex', 'dt', 'time_stayed', 'outcome', 'datsym', 'age'],
                                 columns_to_readd_at_end=['sex', 'age', 'outcome']):
    """ Using KNN algorithm to predict missing values

    Parameters
    ----------
    X: matrix of features

    Returns
    -------
    
    - A plot with correlation features
    
    """
    rest = subtract_list(data_frame.columns, no_need_duplicate_columns)
    df_to_rearange = data_frame[rest].copy()
    df_to_rearange['time_elapsed'] = df_to_rearange[current_time_column] - df_to_rearange[first_day_column]
    df_rearanged = rearange_horizontally(df_to_rearange, 'time_elapsed', reference_column)
    df_rearanged = df_rearanged.reset_index()
    df_rearanged = df_rearanged.rename(index=str, columns={'index':'msfid'})
    df_tail = data_frame[[reference_column] + columns_to_readd_at_end].copy()
    df_tail_shrunk = df_tail.groupby(reference_column).nth(0)
    df_rearanged_with_end = pd.merge(df_rearanged.set_index(reference_column), df_tail_shrunk, left_index=True, right_index=True, how='inner')
    return df_rearanged_with_end.reset_index()


########################################### DATA - VISUALIZATION ##############################################################



def Corr_vision(X):
    """ Correlation visualization according to Pearson

    Parameters
    ----------
    X: matrix of features

    Returns
    -------
    
    - A plot with correlation features
    
    """


    fig, ax = plt.subplots(figsize=(20,20))
    visualizer = Rank2D(algorithm="pearson")
    visualizer.fit_transform(X)
    #visualizer.show('corr_matrix') // to output png
    plt.show()

def Imbalance(y):
    """ Imabalance between the labels

    Parameters
    ----------
    y: vector of labels

    Returns
    -------
    
    - A plot with the class imbalances for Ebola positive or negative
    
    """
    # Instantiate the visualizer
    visualizer = ClassBalance(labels=['Ebola negative', 'Ebola positive'])

    visualizer.fit(y)                        # Fit the data to the visualizer
    #visualizer.show('class_balance')        # Finalize and render the figure
    plt.show()

def Imbalance_out(y):
    """ Imabalance between the labels

    Parameters
    ----------
    y: vector of labels

    Returns
    -------
    
    - A plot with the class imbalances for the outcome
    
    """
    # Instantiate the visualizer
    visualizer = ClassBalance(labels=[ 'Survival','Death'])

    visualizer.fit(y)                        # Fit the data to the visualizer
    #visualizer.show('class_balance')        # Finalize and render the figure
    plt.show()
    
def Rad_vision(X,y):
    """ Radial distrubtions of cases around the systems, a method to detect separability between classes

    Parameters
    ----------
    X: matrix of features
    
    y: vector of labels, for diagnosis

    Returns
    -------
    
    - A radial plot, with the labels and the features at the circonference
    
    """

    fig, ax = plt.subplots(figsize=(20,10))
    radviz(X, y.values, classes = ['Ebola negative', 'Ebola positive'])
    plt.show()

def Rad_vision_out(X,y):
    """ Radial distrubtions of cases around the systems, a method to detect separability between classes

    Parameters
    ----------
    X: matrix of features
    
    y: vector of labels, for prognosis

    Returns
    -------
    
    - A radial plot, with the labels and the features at the circonference
    
    """
    fig, ax = plt.subplots(figsize=(20,10))
    radviz(X, y.values, classes = ['Survival', 'Death'])
    plt.show()
    

def score_model(X_train, y_train, X_test, y_test, model,  **kwargs):
    """ A function that returns the different metrics of accuracy, confusion matrix and other model reports depending on the type of model that is asked.
    
    This function is for diagnosis, please use score_model_outcome for prognosis

    Parameters
    ----------
    X_train: matrix of training features
    
    y_train: vector of training labels
    
    X_test: matrix of test features
    
    y_test: vector of test labels

    Returns
    -------
    
    - Accuracy, F1 score and ROC_AUC for the train and test set
    
    - Confusion matrix
    
    - ClassificationReport
    
    - PrecisionRecallCurve
    
    - ClassPredictionError
    
    """

    
    # Train the model
    model.fit(X_train, y_train, **kwargs)
    
    # Predict on the train set
    prediction_train = model.predict(X_train)
    
    # Compute metrics for the train set
    accuracy_train = accuracy_score(y_train, prediction_train)
    
    #False Positive Rate, True Positive Rate, Threshold
    fpr_train, tpr_train, thresholds_train = roc_curve(y_train, prediction_train)
    auc_train = auc(fpr_train, tpr_train)
    
    f1_score_train = f1_score(y_train, prediction_train)

    # Predict on the test set
    prediction_test = model.predict(X_test)
    
    accuracy_test = accuracy_score(y_test, prediction_test)

    fpr_test, tpr_test, thresholds_test = roc_curve(y_test, prediction_test)
    auc_test = auc(fpr_test, tpr_test)
    
    f1_score_test = f1_score(y_test, prediction_test)
    
    print("{}:".format(model.__class__.__name__))
    # Compute and return F1 (harmonic mean of precision and recall)
    print("On training we get an Accuracy {}, an AUC {} and F1 score {} ".format(accuracy_train, auc_train, f1_score_train ) )
    
    print("For test we get an Accuracy {}, an AUC {} and F1 score {}".format(accuracy_test, auc_test, f1_score_test) )
    
    fig, axes = plt.subplots(3, 2, figsize = (20,20))

    visualgrid = [
    ConfusionMatrix(model, ax=axes[0][0], classes=['Ebola Negative', 'Ebola Positive'], cmap="YlGnBu"),
    ClassificationReport(model, ax=axes[0][1], classes=['Ebola Negative', 'Ebola Positive'],cmap="YlGn",),
    PrecisionRecallCurve(model, ax=axes[1][0]),
    ClassPredictionError(model, classes=['Ebola Negative', 'Ebola Positive'], ax=axes[1][1]),
    ]

    for viz in visualgrid:
        viz.fit(X_train, y_train)
        viz.score(X_test, y_test)
        viz.finalize()
    
    try:
        roc_auc(model, X_train, y_train, X_test=X_test, y_test=y_test, classes=['Ebola Negative', 'Ebola Positive'], ax=axes[2][0])
    except:
        print('Can plot ROC curve for this model')
    
    try:
        viz = FeatureImportances(model,ax=axes[2][1], stack=True, relative=False)
        viz.fit(X_train, y_train)
        viz.score(X_test, y_test)
        viz.finalize()
    except:
        print('Don\'t have feature importance')
        
    plt.show()
    print('\n')
    
    
    
    

def score_model_outcome(X_train, y_train, X_test, y_test, model,  **kwargs):
    """ A function that returns the different metrics of accuracy, confusion matrix and other model reports depending on the type of model that is asked.
    
    This function is for prognosis

    Parameters
    ----------
    X_train: matrix of training features
    
    y_train: vector of training labels
    
    X_test: matrix of test features
    
    y_test: vector of test labels

    Returns
    -------
    
    - Accuracy, F1 score and ROC_AUC for the train and test set
    
    - Confusion matrix
    
    - ClassificationReport
    
    - PrecisionRecallCurve
    
    - ClassPredictionError
    
    """
    
    # Train the model
    model.fit(X_train, y_train, **kwargs)
    
    # Predict on the train set
    prediction_train = model.predict(X_train)
    
    # Compute metrics for the train set
    accuracy_train = accuracy_score(y_train, prediction_train)
    
    #False Positive Rate, True Positive Rate, Threshold
    fpr_train, tpr_train, thresholds_train = roc_curve(y_train, prediction_train)
    auc_train = auc(fpr_train, tpr_train)
    
    f1_score_train = f1_score(y_train, prediction_train)

    # Predict on the test set
    prediction_test = model.predict(X_test)
    
    accuracy_test = accuracy_score(y_test, prediction_test)
    

    fpr_test, tpr_test, thresholds_test = roc_curve(y_test, prediction_test)
    auc_test = auc(fpr_test, tpr_test)
    
    f1_score_test = f1_score(y_test, prediction_test)
    
    print("{}:".format(model.__class__.__name__))
    # Compute and return F1 (harmonic mean of precision and recall)
    print("On training we get an Accuracy {}, an AUC {} and F1 score {} ".format(accuracy_train, auc_train, f1_score_train ) )
    
    print("For test we get an Accuracy {}, an AUC {} and F1 score {}".format(accuracy_test, auc_test, f1_score_test) )
    
    fig, axes = plt.subplots(3, 2, figsize = (20,20))


    visualgrid = [
    ConfusionMatrix(model, ax=axes[0][0], classes=['Death', 'Survival'], cmap="YlGnBu"),
    ClassificationReport(model, ax=axes[0][1], classes=['Death', 'Survival'],cmap="YlGn",),
    PrecisionRecallCurve(model, ax=axes[1][0]),
    ClassPredictionError(model, classes=['Death', 'Survival'], ax=axes[1][1]),
    ]

    for viz in visualgrid:
        viz.fit(X_train, y_train)
        viz.score(X_test, y_test)
        viz.finalize()
    
    try:
        roc_auc(model, X_train, y_train, X_test=X_test, y_test=y_test, classes=['Death', 'Survival'], ax=axes[2][0])
    except:
        print('Can plot ROC curve for this model')
    
    try:
        viz = FeatureImportances(model,ax=axes[2][1], stack=True, relative=False)
        viz.fit(X_train, y_train)
        viz.score(X_test, y_test)
        viz.finalize()
    except:
        print('Don\'t have feature importance')
        
    plt.show()
    print('\n')



def PCA_vision_3D(X,y):
    visualizer = PCA_3D(scale=True, projection=3, classes=['Ebola negative', 'Ebola positive'])
    visualizer.fit_transform(X, y)
    #visualizer.show()
    plt.show()

def PCA_vision_3D_out(X,y):
    visualizer = PCA_3D(scale=True, projection=3, classes=['Survival', 'Death'])
    visualizer.fit_transform(X, y)
    #visualizer.show()
    plt.show()



    
def Random_forest(X_train,y_train,n_est, index_tree,max_depth,sample_split):
    """Perform random forest.

    Parameters
    ----------
    X_train: matrix of features (training set)
    y_train: Labels (training set)
    
    n_est: number of estimators (aka number of trees)
    
    sample_split: minimum number of sample to require a split (aka min_samples_split)
    
    split: default = 0.3, percentage used for testing in split_data function
    
    index_tree : Which decision tree to visualize
        
    Returns
    -------
    - 
    
    """

    clf = RandomForestClassifier(n_estimators=n_est, max_depth=max_depth, min_samples_split = sample_split, random_state = 123)
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)

    print(clf.estimators_[index_tree])
    
    accuracy_all(y_test, y_pred)

    dot_data_forest =export_graphviz(
    clf.estimators_[2],
    out_file=None,
    feature_names=X.columns,
    class_names=['Not a case', 'Confirmed'],
    label='root',
    filled=True,
    rounded=True,
    impurity=False,
    proportion=True)
    graph_forest = graphviz.Source(dot_data_forest)
    #graph_forest.render("Decision_forest") // output 


    
     
