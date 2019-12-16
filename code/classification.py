#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 21:15:15 2019

@author: Pak Shing Ho
"""

import pandas as pd

"""
######################################################
# Part 1. Specifications: File Import Settings,      #
                          Element Name Choices, and  #
                          Occupation Label Choices   #
######################################################
"""

# Parameters to read files
index_col = 'O*NET-SOC Code'
usecols = ['O*NET-SOC Code', 'Element Name', 'Scale ID', 'Data Value'] # data columns used

# Element Name to consider as features
skills = ['Critical Thinking', 
          'Complex Problem Solving',
          'Active Learning',
          'Judgment and Decision Making']

abilities = ['Deductive Reasoning',
             'Inductive Reasoning',
             'Information Ordering',
             'Mathematical Reasoning']

work_styles = ['Analytical Thinking',
               'Innovation',
               'Attention to Detail']

work_activities = ['Analyzing Data or Information',
                   'Making Decisions and Solving Problems',
                   'Thinking Creatively',
                   'Performing General Physical Activities']

# list of hand-labeled analytical occupations (45)
analytical = list(set(
             ['Logistics Analysts',
              'Management Analysts',
              'Market Research Analysts and Marketing Specialists',
              'Budget Analysts',
              'Credit Analysts',
              'Financial Analysts',
              'Financial Quantitative Analysts',
              'Fraud Examiners, Investigators and Analysts',
              'Computer Systems Analysts',
              'Information Security Analysts',
              'Business Intelligence Analysts',
              'Operations Research Analysts',
              'Climate Change Analysts',
              'Quality Control Analysts',
              'Intelligence Analysts',
              'Risk Management Specialists',
              'Political Scientists',
              'Computer and Information Research Scientists',
              'Computer Network Architects',
              'Search Marketing Strategists',
              'Mathematicians',
              'Animal Scientists',
              'Soil and Plant Scientists',
              'Biochemists and Biophysicists',
              'Zoologists and Wildlife Biologists',
              'Bioinformatics Scientists',
              'Medical Scientists, Except Epidemiologists',
              'Physicists',
              'Economists',
              'Computer Science Teachers, Postsecondary',
              'Engineering Teachers, Postsecondary',
              'Physics Teachers, Postsecondary',
              'Aerospace Engineers',
              'Biomedical Engineers',
              'Civil Engineers',
              'Electrical Engineers',
              'Mechanical Engineers',
              'Chemical Engineers',
              'Nuclear Engineers',
              'Actuaries',
              'Software Developers, Applications',
              'Accountants',
              'Lawyers',
              'Judges, Magistrate Judges, and Magistrates',
              'Clinical Psychologists'
              ]))

# list of hand-labeled non-analytical occupations (45)
non_analytical = list(set(
                 ['Bus Drivers, Transit and Intercity',
                  'Bus Drivers, School or Special Client',
                  'Driver/Sales Workers',
                  'Heavy and Tractor-Trailer Truck Drivers',
                  'Light Truck or Delivery Services Drivers',
                  'Taxi Drivers and Chauffeurs',
                  'Janitors and Cleaners, Except Maids and Housekeeping Cleaners',
                  'Maids and Housekeeping Cleaners',
                  'Cooks, Fast Food',
                  'Cooks, Institution and Cafeteria',
                  'Cooks, Private Household',
                  'Cooks, Restaurant',
                  'Food Preparation Workers',
                  'Bartenders',
                  'Waiters and Waitresses',
                  'Food Servers, Nonrestaurant',
                  'Dining Room and Cafeteria Attendants and Bartender Helpers',
                  'Dishwashers',
                  'Barbers',
                  'Models',
                  'Singers',
                  'Dancers',
                  'Actors',
                  'Cashiers',
                  'Shampooers',
                  'Telemarketers',
                  'Door-To-Door Sales Workers, News and Street Vendors, and Related Workers',
                  'Tapers',
                  'Fishers and Related Fishing Workers',
                  'Farmworkers and Laborers, Crop',
                  'Sewers, Hand',
                  'Print Binding and Finishing Workers',
                  'Bakers',
                  'Butchers and Meat Cutters',
                  'Meat, Poultry, and Fish Cutters and Trimmers',
                  'Slaughterers and Meat Packers',
                  'Food Batchmakers',
                  'Graders and Sorters, Agricultural Products',
                  'Cutters and Trimmers, Hand',
                  'Sewing Machine Operators',
                  'Shoe and Leather Workers and Repairers',
                  'Laborers and Freight, Stock, and Material Movers, Hand',
                  'Painting, Coating, and Decorating Workers',
                  'Laundry and Dry-Cleaning Workers',
                  'Cleaners of Vehicles and Equipment'
                  ]))


"""
################################################
# Part 2. Read data files and collect features #
################################################
"""
# Read Occupation Data file
Occupation = pd.read_excel(io='../db_24_1_excel/Occupation Data.xlsx',
                           index_col=index_col,
                           usecols=['O*NET-SOC Code','Title'])

Occupation.loc[Occupation['Title'].isin(analytical), 'Analytical'] = 1
Occupation.loc[Occupation['Title'].isin(non_analytical), 'Analytical'] = 0


# Read Skills file and extract skills features
Skills = pd.read_excel(io='../db_24_1_excel/Skills.xlsx',
                       index_col=index_col,
                       usecols=usecols)

Skills = Skills[(Skills['Scale ID']=='LV') & (Skills['Element Name'].isin(skills))]
Skills.drop(columns='Scale ID', inplace=True)
Skills = Skills.pivot(columns='Element Name', values='Data Value') # Reshape dataframe From long format to wide format


# Read Abilities file and extract abilities features
Abilities = pd.read_excel(io='../db_24_1_excel/Abilities.xlsx',
                          index_col=index_col,
                          usecols=usecols)

Abilities = Abilities[(Abilities['Scale ID']=='LV') & (Abilities['Element Name'].isin(abilities))]
Abilities.drop(columns='Scale ID', inplace=True)
Abilities = Abilities.pivot(columns='Element Name', values='Data Value') # Reshape dataframe From long format to wide format


# Read Work Styles file and extract work styles features
Work_Styles = pd.read_excel(io='../db_24_1_excel/Work Styles.xlsx',
                          index_col=index_col,
                          usecols=usecols)

Work_Styles = Work_Styles[(Work_Styles['Scale ID']=='IM') & (Work_Styles['Element Name'].isin(work_styles))]
Work_Styles.drop(columns='Scale ID', inplace=True)
Work_Styles = Work_Styles.pivot(columns='Element Name', values='Data Value') # Reshape dataframe From long format to wide format


# Read Work Activities file and extract work activities features
Work_Activities = pd.read_excel(io='../db_24_1_excel/Work Activities.xlsx',
                          index_col=index_col,
                          usecols=usecols)

Work_Activities = Work_Activities[(Work_Activities['Scale ID']=='LV') & (Work_Activities['Element Name'].isin(work_activities))]
Work_Activities.drop(columns='Scale ID', inplace=True)
Work_Activities = Work_Activities.pivot(columns='Element Name', values='Data Value') # Reshape dataframe From long format to wide format


# Merge cleaned dataframes into one dataframe
merged = Occupation.join(Skills, how='inner').join(Abilities, how='inner').join(Work_Styles, how='inner').join(Work_Activities, how='inner')

# Examine correlation
correlation=merged.corr()
# pd.plotting.scatter_matrix(merged) # plot scatter matrix

"""
###################################################
# Part 3. Train models with different classifiers #
###################################################
"""

# Divide data set into training and testing sets
train = merged[merged['Analytical'].notna()]
test = merged[merged['Analytical'].isna()]
train_X = train.iloc[:,2:]
train_Y = train['Analytical']
test_X = test.iloc[:,2:]
full_X = merged.iloc[:,2:]
full_result = merged.copy()


# Decsion Tree (the worst choice since we are doing binary classification with more than one feature)
from sklearn import tree
clf_tree = tree.DecisionTreeClassifier(random_state=2).fit(train_X, train_Y)
full_result['Tree'] = clf_tree.predict(full_X) # add prediction to dataframe with feature columns
#tree.plot_tree(clf_tree.fit(train_X, train_Y)) 


# Logistic Regression
from sklearn.linear_model import LogisticRegression
Logistic = LogisticRegression(random_state=0,
                                  penalty='none').fit(train_X, train_Y)
full_result['LR'] = Logistic.predict(full_X) # add prediction to dataframe with feature columns
full_result['LR_prob'] = Logistic.predict_proba(full_X)[:,1] # add probability to dataframe with feature columns


# Linear Discriminant Analysis
# =============================================================================
# When the classes are well-separated, the parameter estimates for the
# logistic regression model are surprisingly unstable. Linear discriminant
# analysis does not suffer from this problem.
# =============================================================================
# =============================================================================
# If n is small and the distribution of the predictors X is approximately
# normal in each of the classes, the linear discriminant model is again
# more stable than the logistic regression model.
# =============================================================================

# merged.hist() # plot histograms
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
LDA = LinearDiscriminantAnalysis().fit(train_X, train_Y)
full_result['LDA'] = LDA.predict(full_X)
full_result['LDA_prob'] = LDA.predict_proba(full_X)[:,1]


# K-Nearest Neighbors
# =============================================================================
# KNN is a completely non-parametric approach:
# No assumptions are made about the shape of the decision boundary. Therefore,
# we can expect this approach to dominate LDA and logistic regression
# when the decision boundary is highly non-linear. On the other hand, KNN
# does not tell us which predictors are important; We don’t get a table of
# coefficients.
# =============================================================================
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors=10, weights='distance').fit(train_X, train_Y)
full_result['KNN'] = KNN.predict(full_X)
full_result['KNN_prob'] = KNN.predict_proba(full_X)[:,1]


# Gaussian Process Classifier (RBF Kernel, allow non-linear decision boundary)
from sklearn.gaussian_process import GaussianProcessClassifier
GPC = GaussianProcessClassifier(random_state=0).fit(train_X, train_Y)
full_result['GPC'] = GPC.predict(full_X)
full_result['GPC_prob'] = GPC.predict_proba(full_X)[:,1]


# SVM (Linear Kernel, good with small training size, take cares of outliers better)
from sklearn import svm
SVM_lin = svm.SVC(kernel='linear', probability=True).fit(train_X, train_Y)
full_result['SVM_lin'] = SVM_lin.predict(full_X)
full_result['SVM_lin_prob'] = SVM_lin.predict_proba(full_X)[:,1]


# SVM (RBF Kernel, good with small training size, take cares of outliers better, allow non-linear decision boundary)
from sklearn import svm
SVM_RBF = svm.SVC(probability=True).fit(train_X, train_Y)
full_result['SVM_RBF'] = SVM_RBF.predict(full_X)
full_result['SVM_RBF_prob'] = SVM_RBF.predict_proba(full_X)[:,1]

# Naive Bayes (Gaussian) (works well with small datasets, conditional independence (i.e. all input features are independent from one another) assumption rarely holds true)
from sklearn.naive_bayes import GaussianNB
GNB = GaussianNB().fit(train_X, train_Y)
full_result['GNB'] = GNB.predict(full_X)
full_result['GNB_prob'] = GNB.predict_proba(full_X)[:,1]


# Random Forest (No assumptions on distribution of data, Handles colinearity better than LR, May not be good with small training data size)
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(random_state=0).fit(train_X, train_Y)
full_result['RF'] = RF.predict(full_X)
full_result['RF_prob'] = RF.predict_proba(full_X)[:,1]


# AdaBoost (No assumptions on distribution of data, Handles colinearity better than LR, May not be good with small training data size)
from sklearn.ensemble import AdaBoostClassifier
Ada = AdaBoostClassifier(random_state=0).fit(train_X, train_Y)
full_result['Ada'] = Ada.predict(full_X)
full_result['Ada_prob'] = Ada.predict_proba(full_X)[:,1]


# Neural Network (Not good when training data is small)
from sklearn.neural_network import MLPClassifier
MLP = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5,2), 
                    random_state=0, 
                    activation='logistic').fit(train_X, train_Y)
full_result['MLP'] = MLP.predict(full_X)
full_result['MLP_prob'] = MLP.predict_proba(full_X)[:,1]


# Voting Classifiers (Hard and Soft Votting)
# =============================================================================
# The idea behind the Voting Classifier is to combine conceptually different 
# machine learning classifiers and use a majority vote or the average predicted
# probabilities (soft vote) to predict the class labels. Such a classifier can 
# be useful for a set of equally well performing model in order to balance out 
# their individual weaknesses.
# =============================================================================
from sklearn.ensemble import VotingClassifier

# estimators to be voted and weighted
estimators=[('LR', Logistic), 
            ('LDA', LDA), 
            ('KNN', KNN),
            ('GPC', GPC),
            ('SVM_RBF', SVM_RBF),
            ('SVM_lin', SVM_lin),
            ('GNB', GNB),
            ('RF', RF)]

# Vote by majority predicted results of estimators considered.
Vote_hard = VotingClassifier(
        estimators=estimators,
        voting='hard').fit(train_X, train_Y)
full_result['Vote_hard'] = Vote_hard.predict(full_X)

# Vote by averaging predicted probabilities of estimators considered
Vote_soft = VotingClassifier(
        estimators=estimators,
        voting='soft').fit(train_X, train_Y)
full_result['Vote_soft'] = Vote_soft.predict(full_X)
full_result['Vote_soft_prob'] = Vote_soft.predict_proba(full_X)[:,1]


# Keep only the estimated results of various estimators
model_result = full_result[['Title', 'Analytical', 'Tree',
       'LR', 'LR_prob', 'LDA', 'LDA_prob', 'KNN', 'KNN_prob', 'GPC',
       'GPC_prob', 'SVM_RBF', 'SVM_RBF_prob', 'SVM_lin', 'SVM_lin_prob', 'GNB',
       'GNB_prob', 'RF', 'RF_prob', 'Ada', 'Ada_prob', 'MLP', 'MLP_prob',
       'Vote_hard', 'Vote_soft', 'Vote_soft_prob']]
