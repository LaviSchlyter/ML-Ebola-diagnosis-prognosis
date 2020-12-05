# -*- coding: utf-8 -*-
"""Some helper functions for project 2."""
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from yellowbrick.features import PCA as PCA_3D
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from sklearn import tree
import graphviz
import statsmodels.api as sm
from sklearn import svm
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LassoCV
from yellowbrick.features import Rank2D
from yellowbrick.target import ClassBalance
from yellowbrick.features.radviz import radviz

    
########################################### FEATURE - ENGINEERING ############################################################

def delete_minus1(X): 
    """ Basic cleaning function 
    If over 50 % missing within a feature, remove the feature 
    and then remove all rows with missing values
    """
    Xp = X.copy()
    k =0;

    for column in Xp:

        if sum(Xp[column]==-1) >= 0.5 * len(Xp[column]):
            Xp.drop(column, axis=1, inplace = True)
            continue

        
        index_names = Xp[(Xp[column] == -1)].index;
        Xp.drop(index_names, inplace = True)
    
    return Xp


# Makes a column full of ones to the right, with the same name as the target column and suffix _indicator (makes copy of dataFrame)
# If the column already exists: does nothing

def make_ones_indicator_column(data_frame, name_of_column_target, inplace=False):
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


        
        
# Finds in the indicator column the lines where the target column has target value, and puts 0 there

def put_zero_in_indicator_column(data_frame, name_of_column_target, target_value, inplace=False):
    if inplace:
        data_frame.loc[data_frame[name_of_column_target] == target_value, name_of_column_target + '_indicator'] = 0
    else :
        df_temp = data_frame.copy()
        df_temp.loc[df_temp[name_of_column_target] == target_value, name_of_column_target + '_indicator'] = 0
        return df_temp
    



# adds a column to the right, with name of target + _indicator, with 0 on same line as the target column has target value

def make_indicator_for_bad_data(data_frame, name_of_column_target, target_value, inplace=False):
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
    """Using KNN algorithm to predict missing values"""
    imputer = SimpleImputer(missing_values= target_values)
    mod = imputer.fit(X)
    X_trans = pd.DataFrame(mod.transform(X))
    X_trans.columns = X.columns
    return X_trans


# To transform the format MM/DD/YYYY or M/DD/YYYY or MM.DD.YYYY into a single integer
def date_transformer(date_in_str):
    day  = ''
    month= ''
    year = ''
    separator = 0
    for i in date_in_str:
        if i =='.' or i=='/':
            separator +=1
        else:
            if separator == 0:
                month += i
            elif separator == 1:
                day   += i
            elif separator ==2:
                year  += i
    return int(day)*10 + int(month)*1000 + int(year)*100000


#  Reformats the date in the desired column using date_transformer
def transform_datclin(data_frame, target_column_name='datclin', inplace=False):
    df = data_frame
    if not inplace:
        df = data_frame.copy()
    for i, date in enumerate(df[target_column_name]):
        nb_date = date_transformer(date)
        df.loc[i,target_column_name] = nb_date
    if not inplace:
        return df


# to extract just one subset of data:
def extract_certain_dataset(data_frame, name_of_target_column, target_value):
    df = data_frame[data_frame[name_of_target_column] == target_value].copy()
    return df


# to separate the dataframe in subgroups depending on the values in the target column.
def make_list_by_value(data_frame, name_of_target_column, name_of_reference_column):
    list_of_df = []
    list_of_target_values = data_frame[name_of_target_column].value_counts().index
    for i, value in enumerate(list_of_target_values):
        list_of_df.append(extract_certain_dataset(data_frame, name_of_target_column, value).set_index(name_of_reference_column))
    return list_of_df


# first separates the original dataset into subgroups corresponding to values on the target column,
# and then puts them back toghether, but horizontally intead of vertically. Aloigned around the reference column.
def rearange_horizontally(data_frame, name_of_target_column, name_of_reference_column):
    list_of_df = make_list_by_value(data_frame, name_of_target_column, name_of_reference_column)
    return pd.concat(list_of_df, axis=1, sort=False).reset_index()
    

# ouputs a list of sub-dataframes. One for each different value in the targeted column 
# (slightly different from make_list_by_value because of context of use)
def dissassemble(data_frame, name_of_column_target):
    out_list=[]
    for i, value in enumerate(data_frame[name_of_column_target].value_counts(sort=False).index):
        df = data_frame[data_frame[name_of_column_target] == value].copy()
        out_list.append(df)
    return out_list


# outputs a list of lists of sub-dataframes. (calls dissassemble twice)
def fine_dissassembly(data_frame, name_first_column_target, name_second_column_target):
    out_list = dissassemble(data_frame, name_first_column_target)
    for i, subdf in enumerate(out_list):
        out_list[i] = dissassemble(subdf, name_second_column_target)
    return out_list


# Undoes dissassemble
def reassemble(list_of_df):
    return pd.concat(list_of_df)


# Undoes fine_dissassembly
def big_reassembly(list_of_list_of_df):
    for i, sublist in enumerate(list_of_list_of_df):
        list_of_list_of_df[i] = reassemble(sublist)
    return reassemble(list_of_list_of_df)


# only works for this specific dataset so many parameters are hard-coded
# will find how many visits each patient did and assign a chronological order to them under a new column 'rank'
def rank_visits_rigorous(data_frame):
    df_clinic_d = transform_datclin(data_frame, 'datclin')      # correct the dates to better suit ranking
    list_of_df = dissassemble(df_clinic_d, 'msfid')             # make subgroups for each patient
    for i, sub_df in enumerate(list_of_df):
        # for each patient, each day of visit, rank the hours of visit of that day
        sub_df['rank_on_day'] = sub_df.groupby('datclin')['timclin'].rank('dense', ascending=True)
    
    df_clinic_m = reassemble(list_of_df)                        # reassemble the dataset
    # combine ranking on day with general date to have exact order of visit
    df_clinic_m['datclin'] = df_clinic_m['datclin'] + df_clinic_m['rank_on_day'] 
    df_clinic_m.drop(columns=['rank_on_day'], inplace=True)
    # find that order
    df_clinic_m['rank']= df_clinic_m.groupby('msfid')['datclin'].rank('dense', ascending=True)
    
    return df_clinic_m
    


########################################### DATA - VISUALIZATION ##############################################################



def Corr_vision(X):
    """ Correlation visualization according to Pearson 
    
    """

    fig, ax = plt.subplots(figsize=(20,10))
    visualizer = Rank2D(algorithm="pearson")
    visualizer.fit_transform(X)
    visualizer.show('corr_matrix')
    plt.show()

def Imbalance(y):
    # Instantiate the visualizer
    visualizer = ClassBalance(labels=['Ebola negative', 'Ebola positive'])

    visualizer.fit(y)                        # Fit the data to the visualizer
    visualizer.show('class_balance')        # Finalize and render the figure
    plt.show()
    
    
def Rad_vision(X,y):
    """ Radial distrubtions of cases around the systems, here change when unknown will be either pos or neg (just like above, also change)
    
    """

    fig, ax = plt.subplots(figsize=(20,10))
    radviz(X, y, classes = ['Ebola negative', 'Ebola positive'])
    plt.show()





def Log_reg(X,y):
    """ Logistic regression, using the p_values to find important features 
    
    A strange thing is Ridha used lgistic regression to find how many features we should keep and then used linear regression with the number of features to 
    find which ones ... ? 
    
    """
    
    #no of features 
    nof_list=np.arange(1, len(X.columns)+1)   

    highest_score=0
    #Variable to store the optimum features
    nof=0           
    score_list =[]
    for n in range(len(nof_list)):
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 123)
        model = LogisticRegression(max_iter=100)
        rfe = RFE(model,n_features_to_select=nof_list[n])
        X_train_rfe = rfe.fit_transform(X_train,y_train)
        X_test_rfe = rfe.transform(X_test)
        model.fit(X_train_rfe,y_train)
        score = model.score(X_test_rfe,y_test)
        score_list.append(score)
        if(score>highest_score):
            highest_score = score
            nof = nof_list[n]
    print("Optimum number of features: %d" %nof)
    print("Score with %d features: %f" % (nof, highest_score))
    cols = np.array(X.columns)
    print(cols[rfe.support_])
    

    print("Feature importance for linear regression")
    
    model = LinearRegression()

    #Initializing RFE model
    rfe = RFE(model, n_features_to_select=nof)   

    #Transforming data using RFE
    X_rfe = rfe.fit_transform(X,y)  

    #Fitting the data to model
    model.fit(X_rfe,y) 

    print(cols[rfe.support_])
    
    
    
    

def backward_elimation(X, y, model_str):
    """Backward Elimination for chosen model, call with backward_elimation(X, y, "least-squares"), selects features """
    cols = list(X.columns)
    pmax = 1
    while (len(cols)>0):
        p= []
        X_1 = X[cols]
        #X_1 = sm.add_constant(X_1) We already added a constant for the model 
        if model_str == "least-squares":
            model = sm.OLS(y,X_1).fit()
        elif model_str == "logistic regression":
            model = sm.Logit(y,X_1).fit()
        else: raise NameError('Backward elimination not implemented for this model')
        p = pd.Series(model.pvalues.values,index = cols)    #[1:]  
        pmax = max(p)
        feature_with_p_max = p.idxmax()
        if(pmax>0.05):
            cols.remove(feature_with_p_max)
        else:
            break
    selected_features_BE = cols
    print(selected_features_BE)
    
    
def Lasso(X,y):
    """ Importance of features according to Lasso method 
    
    """
    reg = LassoCV()
    reg.fit(X, y)

    print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
    print("Best score using built-in LassoCV: %f" %reg.score(X,y))
    coef = pd.Series(reg.coef_, index = X.columns)
    print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
   
    imp_coef = coef.sort_values()
    #import matplotlib
    #matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
    plt.rcParams['figure.figsize'] = (8.0, 10.0)
    imp_coef.plot(kind = "barh")
    plt.title("Feature importance using Lasso Model")
    plt.show()


######################################################### MODELS ######################################################################################

def PCA_(k, X, y):

    pca = PCA(n_components = k)
    X_new = pca.fit_transform(X)
    y_new = y.copy()
    
    coeff = np.transpose(pca.components_[0:2, :])
    labels = None
    xs = X_new[:,0]
    ys = X_new[:,1]
    n = coeff.shape[0]
    cdict = {'0': 'gray', '1': 'black'}
    ldict = {'0': 'Not a case', '1': 'Confirmed'}
    fig, ax = plt.subplots()
    for g in np.unique(y_new):
        ix = np.where(y_new == g)
        ax.scatter(xs[ix], ys[ix], c = cdict[g], label = ldict[g])
    for i in range(n):
        factor_ = 2.8
        plt.arrow(0, 0, coeff[i,0]*factor_, coeff[i,1]*factor_,color = 'r',alpha = 0.5)
        if labels is None and np.linalg.norm([coeff[i,0], coeff[i,1]])>0.3:
            plt.text(coeff[i,0]* 3, coeff[i,1] * 3, str(X.columns[i]), color = 'red', ha = 'center', va = 'center', fontsize=12, weight='bold')
    
    ax.legend()
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid()
    plt.show()
    
    #Call the function. Use only the 2 PCs.
    #myplot(X_new, y_new, np.transpose(pca.components_[0:2, :]))
    
    



def PCA_vision_3D(X,y):
    visualizer = PCA_3D(scale=True, projection=3, classes=['Ebola negative', 'Ebola positive'])
    visualizer.fit_transform(X, y)
    visualizer.show()
    plt.show()





def Decision_trees(X,y,max_depth, split = 0.3):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = split, random_state = 123)
    clfq = tree.DecisionTreeClassifier(max_depth=max_depth)
    clfq = clfq.fit(X_train, y_train)

    y_pred = clfq.predict(X_test)
    dot_data_1q = tree.export_graphviz(clfq, 
                                out_file=None, 
                                feature_names=X.columns,
                                rounded=True)
    graph_1q = graphviz.Source(dot_data_1q)
    graph_1q.render("Decision_tree_1_q")

    accuracy_all(y_test, y_pred)

    
    tree.plot_tree(clfq)
    fig = plt.gcf()
    plt.show()


    


    
def Random_forest(X,y,n_est, index_tree,max_depth,sample_split, split = 0.3):
    """Perform random forest.

    Parameters
    ----------
    X: matrix
        features
    y: label
    
    n_est: number of estimators (aka number of trees)
    
    sample_split: minimum number of sample to require a split (aka min_samples_split)
    
    split: default = 0.3, percentage used for testing in split_data function
    
    index_tree : Which decision tree to visualize
        
    Returns
    -------
    - 
    
    """
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = split, random_state = 123)
    clf = RandomForestClassifier(n_estimators=n_est, max_depth=max_depth, min_samples_split = sample_split, random_state = 123)
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

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
    graph_forest.render("Decision_forest")



    
def accuracy_all(y_test, y_pred):
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("Macro F1 score: ",f1_score(y_test, y_pred, average='macro'))
    print("Micro F1 score: ",f1_score(y_test, y_pred, average='micro'))
    #print("Accuracy under curve: ", roc_auc_score(y_test, y_pred))
    
def elbow_plot(X):

    Sum_of_squared_distances = []
    K = range(1,15)
    for k in K:
        KMeans_clf = KMeans(n_clusters=k).fit(X)
        Sum_of_squared_distances.append(KMeans_clf.inertia_)
    
    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()
    

    


def SVM_(X,y, param_grid = {'C': [0.1,1], 'gamma': [1,0.1],'kernel': ['sigmoid', 'poly']}):
    """
    SVM model
    
    param_grid : by default, but you may override with more values and with sigmoid as well, beware, this is time consuming
    
    
    """
    


    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 123)

    #param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']} (TOO MUCH TIME)
     

    grid = GridSearchCV(svm.SVC(),param_grid,refit=True,verbose=1)
    grid.fit(X_train,y_train)

    print(grid.best_estimator_)

    grid_pred = grid.predict(X_test)
    accuracy_all(grid_pred, y_test)
