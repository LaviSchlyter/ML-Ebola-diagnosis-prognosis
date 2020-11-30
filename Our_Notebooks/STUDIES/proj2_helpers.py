# -*- coding: utf-8 -*-
"""Some helper functions for project 2."""
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV

## PCA

def PCA_(k, X, y):

    pca = PCA(n_components = k)
    X_new = pca.fit_transform(X)
    y_new = y.copy()
    
    labels = None
    xs = X_new[:,0]
    ys = X_new[:,1]
    n = coeff.shape[0]
    cdict = {0: 'gray', 1: 'black'}
    ldict = {0: 'Not a case', 1: 'Confirmed'}
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


def Decision_trees(X,y,max_depth, split = 0.3):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = split, random_state = 123)
    clfq = tree.DecisionTreeClassifier(max_depth=max_depth)
    clfq = clfq.fit(X_train, y_train)

    y_pred = clfq.predict(X_test)
    dot_data_1q = tree.export_graphviz(clf1q, 
                                out_file=None, 
                                feature_names=X.columns,
                                rounded=True)
    graph_1q = graphviz.Source(dot_data_1q)
    graph_1q.render("Decision_tree_1_q")

    accuracy_all(y_test, y_pred)

    
    tree.plot_tree(clfq)
    fig = matplotlib.pyplot.gcf()
    plt.show()

    


    
def Random_forest(X,y,n_est, max_depth,sample_split, split = 0.3):
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = split, random_state = 123)
    clf = RandomForestClassifier(n_estimators=n_est, max_depth=max_depth, min_samples_split = sample_split, random_state = 123)
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    accuracy_all(y_test, y_pred)
"""
    Add this in code to export graph ... with correct estimators_
    dot_data_forest =export_graphviz(
    clf_best.estimators_[0],
    out_file=None,
    feature_names=X.columns,
    class_names=['Not a case', 'Confirmed'],
    label='root',
    filled=True,
    rounded=True,
    impurity=False,
    proportion=True
)
    graph_forest = graphviz.Source(dot_data_forest)
    graph_forest.render("Decision_forest")
"""


    
def accuracy_all(y_test, y_pred):
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("Macro F1 score: ",f1_score(y_test, y_pred, average='macro'))
    print("Micro F1 score: ",f1_score(y_test, y_pred, average='micro'))
    print("Accuracy under curve: ", roc_auc_score(y_test, y_pred))
    
    
