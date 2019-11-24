import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import power_transform
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os
import datetime
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer

#Garbage Collector
import gc

os.getcwd()
os.chdir('C:/Users/Mann-A2/Documents/Python Repository/IEEE Fraud Detection - Kaggle/ieee-fraud-detection')

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


train_identity = pd.read_csv('train_identity.csv')
train_transaction = pd.read_csv('train_transaction.csv')
test_identity = pd.read_csv('test_identity.csv')
test_transaction = pd.read_csv('test_transaction.csv')

# function to reduce size
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

#split function
def id_split(dataframe):
    dataframe['device_name'] = dataframe['DeviceInfo'].str.split('/', expand=True)[0]
    dataframe['device_version'] = dataframe['DeviceInfo'].str.split('/', expand=True)[1]

    dataframe['OS_id_30'] = dataframe['id_30'].str.split(' ', expand=True)[0]
    dataframe['version_id_30'] = dataframe['id_30'].str.split(' ', expand=True)[1]

    dataframe['browser_id_31'] = dataframe['id_31'].str.split(' ', expand=True)[0]
    dataframe['version_id_31'] = dataframe['id_31'].str.split(' ', expand=True)[1]

    dataframe['screen_width'] = dataframe['id_33'].str.split('x', expand=True)[0]
    dataframe['screen_height'] = dataframe['id_33'].str.split('x', expand=True)[1]

    dataframe['id_34'] = dataframe['id_34'].str.split(':', expand=True)[1]
    dataframe['id_23'] = dataframe['id_23'].str.split(':', expand=True)[1]

    dataframe.loc[dataframe['device_name'].str.contains('SM', na=False), 'device_name'] = 'Samsung'
    dataframe.loc[dataframe['device_name'].str.contains('SAMSUNG', na=False), 'device_name'] = 'Samsung'
    dataframe.loc[dataframe['device_name'].str.contains('GT-', na=False), 'device_name'] = 'Samsung'
    dataframe.loc[dataframe['device_name'].str.contains('Moto G', na=False), 'device_name'] = 'Motorola'
    dataframe.loc[dataframe['device_name'].str.contains('Moto', na=False), 'device_name'] = 'Motorola'
    dataframe.loc[dataframe['device_name'].str.contains('moto', na=False), 'device_name'] = 'Motorola'
    dataframe.loc[dataframe['device_name'].str.contains('LG-', na=False), 'device_name'] = 'LG'
    dataframe.loc[dataframe['device_name'].str.contains('rv:', na=False), 'device_name'] = 'RV'
    dataframe.loc[dataframe['device_name'].str.contains('HUAWEI', na=False), 'device_name'] = 'Huawei'
    dataframe.loc[dataframe['device_name'].str.contains('ALE-', na=False), 'device_name'] = 'Huawei'
    dataframe.loc[dataframe['device_name'].str.contains('-L', na=False), 'device_name'] = 'Huawei'
    dataframe.loc[dataframe['device_name'].str.contains('Blade', na=False), 'device_name'] = 'ZTE'
    dataframe.loc[dataframe['device_name'].str.contains('BLADE', na=False), 'device_name'] = 'ZTE'
    dataframe.loc[dataframe['device_name'].str.contains('Linux', na=False), 'device_name'] = 'Linux'
    dataframe.loc[dataframe['device_name'].str.contains('XT', na=False), 'device_name'] = 'Sony'
    dataframe.loc[dataframe['device_name'].str.contains('HTC', na=False), 'device_name'] = 'HTC'
    dataframe.loc[dataframe['device_name'].str.contains('ASUS', na=False), 'device_name'] = 'Asus'

    dataframe.loc[dataframe.device_name.isin(dataframe.device_name.value_counts()[dataframe.device_name.value_counts() < 200].index), 'device_name'] = "Others"
    dataframe['had_id'] = 1
    gc.collect()
    
    return dataframe

train_identity = id_split(train_identity)
test_identity = id_split(test_identity)


#Data joining

train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left', left_index=True, right_index=True)
test = pd.merge(test_transaction, test_identity, on='TransactionID', how='left', left_index=True, right_index=True)

print('Data was successfully merged!\n')

del train_identity, train_transaction, test_identity, test_transaction

print(f'Train dataset has {train.shape[0]} rows and {train.shape[1]} columns.')
print(f'Test dataset has {test.shape[0]} rows and {test.shape[1]} columns.\n')


#============================================================================

useful_features = ['isFraud', 'TransactionDT', 'TransactionAmt', 'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2', 'dist1',
                   'P_emaildomain', 'R_emaildomain', 'C1', 'C2', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13',
                   'C14', 'D1', 'D2', 'D3', 'D4', 'D5', 'D10', 'D11', 'D15', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 
				   'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V17',
                   'V19', 'V20', 'V29', 'V30', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38', 'V40', 'V44', 'V45', 'V46', 'V47', 'V48',
                   'V49', 'V51', 'V52', 'V53', 'V54', 'V56', 'V58', 'V59', 'V60', 'V61', 'V62', 'V63', 'V64', 'V69', 'V70', 'V71',
                   'V72', 'V73', 'V74', 'V75', 'V76', 'V78', 'V80', 'V81', 'V82', 'V83', 'V84', 'V85', 'V87', 'V90', 'V91', 'V92',
                   'V93', 'V94', 'V95', 'V96', 'V97', 'V99', 'V100', 'V126', 'V127', 'V128', 'V130', 'V131', 'V138', 'V139', 'V140',
                   'V143', 'V145', 'V146', 'V147', 'V149', 'V150', 'V151', 'V152', 'V154', 'V156', 'V158', 'V159', 'V160', 'V161',
                   'V162', 'V163', 'V164', 'V165', 'V166', 'V167', 'V169', 'V170', 'V171', 'V172', 'V173', 'V175', 'V176', 'V177',
                   'V178', 'V180', 'V182', 'V184', 'V187', 'V188', 'V189', 'V195', 'V197', 'V200', 'V201', 'V202', 'V203', 'V204',
                   'V205', 'V206', 'V207', 'V208', 'V209', 'V210', 'V212', 'V213', 'V214', 'V215', 'V216', 'V217', 'V219', 'V220',
                   'V221', 'V222', 'V223', 'V224', 'V225', 'V226', 'V227', 'V228', 'V229', 'V231', 'V233', 'V234', 'V238', 'V239',
                   'V242', 'V243', 'V244', 'V245', 'V246', 'V247', 'V249', 'V251', 'V253', 'V256', 'V257', 'V258', 'V259', 'V261',
                   'V262', 'V263', 'V264', 'V265', 'V266', 'V267', 'V268', 'V270', 'V271', 'V272', 'V273', 'V274', 'V275', 'V276',
                   'V277', 'V278', 'V279', 'V280', 'V282', 'V283', 'V285', 'V287', 'V288', 'V289', 'V291', 'V292', 'V294', 'V303',
                   'V304', 'V306', 'V307', 'V308', 'V310', 'V312', 'V313', 'V314', 'V315', 'V317', 'V322', 'V323', 'V324', 'V326',
                   'V329', 'V331', 'V332', 'V333', 'V335', 'V336', 'V338', 'id_01', 'id_02', 'id_05', 'id_06', 
                   'id_11', 'id_12', 'id_13', 'id_15', 'id_17', 'id_19', 'id_20', 'id_31', 'id_36', 'id_37', 'id_38', 'DeviceType', 
				   'DeviceInfo', 'device_name', 'device_version', 'OS_id_30', 'version_id_30',
                   'browser_id_31', 'version_id_31', 'screen_width', 'screen_height', 'had_id']


cols_to_drop = [col for col in train.columns if col not in useful_features]

train = train.drop(cols_to_drop, axis=1)
test = test.drop(cols_to_drop, axis=1)

#Merging email columns
train.P_emaildomain.fillna(train.R_emaildomain, inplace=True)
del train['R_emaildomain']

test.P_emaildomain.fillna(test.R_emaildomain, inplace=True)
del test['R_emaildomain']


# New feature - log of transaction amount. ()
train['TransactionAmt_Log'] = np.log(train['TransactionAmt'])
test['TransactionAmt_Log'] = np.log(test['TransactionAmt'])

# New feature - decimal part of the transaction amount.
train['TransactionAmt_decimal'] = ((train['TransactionAmt'] - train['TransactionAmt'].astype(int)) * 1000).astype(int)
test['TransactionAmt_decimal'] = ((test['TransactionAmt'] - test['TransactionAmt'].astype(int)) * 1000).astype(int)

# New feature - day of week in which a transaction happened.
train['Transaction_day_of_week'] = np.floor((train['TransactionDT'] / (3600 * 24) - 1) % 7)
test['Transaction_day_of_week'] = np.floor((test['TransactionDT'] / (3600 * 24) - 1) % 7)

# New feature - hour of the day in which a transaction happened.
train['Transaction_hour'] = np.floor(train['TransactionDT'] / 3600) % 24
test['Transaction_hour'] = np.floor(test['TransactionDT'] / 3600) % 24


del train['TransactionAmt'], train['TransactionDT']
del test['TransactionAmt'], test['TransactionDT']


#handling missing values -- replacing with -999
train.replace(np.nan, -999, inplace=True)
test.replace(np.nan, -999, inplace=True)

train.isnull().sum()
test.isnull().sum()


#=====================

#You can use isnull with mean for treshold and then remove columns by boolean
# indexing with loc (because remove columns), also need invert condition - 
# so <.8 means remove all columns >=0.8:
#
#df = df.loc[:, df.isnull().mean() < .8]



#Label Encoding

# Encoding - count encoding for both train and test
for feature in ['card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'id_36']:
    train[feature + '_count_full'] = train[feature].map(pd.concat([train[feature], test[feature]], ignore_index=True).value_counts(dropna=False))
    test[feature + '_count_full'] = test[feature].map(pd.concat([train[feature], test[feature]], ignore_index=True).value_counts(dropna=False))

# Encoding - count encoding separately for train and test
for feature in ['id_01', 'id_31', 'id_36']:
    train[feature + '_count_dist'] = train[feature].map(train[feature].value_counts(dropna=False))
    test[feature + '_count_dist'] = test[feature].map(test[feature].value_counts(dropna=False))


for col in train.columns:
    if train[col].dtype == 'object':
        le = LabelEncoder()
        le.fit(list(train[col].astype(str).values) + list(test[col].astype(str).values))
        train[col] = le.transform(list(train[col].astype(str).values))
        test[col] = le.transform(list(test[col].astype(str).values))




train.to_csv('training_set.csv', index = None, header=True)
test.to_csv('testing_set.csv', index = None, header=True)

print("\nData successfully prepared")


#=======================================================================================================================
#Model Building
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import power_transform
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import cohen_kappa_score, make_scorer
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from pystacknet.pystacknet import StackNetClassifier

#pip install lightgbm

#pip install catboost

#pip install pystacknet-master

import matplotlib.pyplot as plt
import os
import datetime
import numpy as np
import seaborn as sns
os.getcwd()
os.chdir('C:/Users/Mann-A2/Documents/Python Repository/IEEE Fraud Detection - Kaggle/ieee-fraud-detection - worked')


train = pd.read_csv('training_set.csv')
test = pd.read_csv('testing_set.csv')

train = reduce_mem_usage(train)
test = reduce_mem_usage(test)


X_train = train.drop(['isFraud'], axis=1)
y_train = train['isFraud']
X_test = test



#LR #0.8052
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=123,solver = 'liblinear')
clf.fit(X_train, y_train)

y_pred = clf.predict_proba(X_test)

pd.DataFrame(y_pred, columns=['predictions','isFraud']).to_csv('prediction LR.csv')


#RF #0.9119
from sklearn.ensemble import RandomForestRegressor

clf = RandomForestRegressor(
    n_estimators=200, max_features=0.4, min_samples_split=50, 
    min_samples_leaf=100, n_jobs=-1, verbose=2)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
pd.DataFrame(y_pred).to_csv('prediction RF.csv')


#Simple -- XGB #0.9342

clf = XGBClassifier(n_estimators=500,
                        n_jobs=4,
                        max_depth=9,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        missing=-999)

clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)
pd.DataFrame(y_pred, columns=['predictions','isFraud']).to_csv('prediction XGB.csv')




#**StackNet Model**-==============


# LGBMClassifier without GPU

clf_lgb = LGBMClassifier(
    max_bin=63,
    num_leaves=255,
    num_iterations=1000,
    learning_rate=0.01,
    tree_learner="serial",
    task="train",
    is_training_metric=False,
    min_data_in_leaf=1,
    min_sum_hessian_in_leaf=100,
    sparse_threshold=1.0,
    num_thread=-1,
    save_binary=True,
    seed=42,
    feature_fraction_seed=42,
    bagging_seed=42,
    drop_seed=42,
    data_random_seed=42,
    objective="binary",
    boosting_type="gbdt",
    verbose=1,
    metric="auc",
    is_unbalance=True,
    boost_from_average=False,
)

clf_lgb.fit(X_train, y_train)
y_pred = clf_lgb.predict_proba(X_test)
pd.DataFrame(y_pred, columns=['predictions','isFraud']).to_csv('prediction LGBMClassifier.csv')


# XGBClassifier without GPU

clf_xgb = XGBClassifier(
    n_estimators=1000,
    max_depth=9,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    missing=-999,
    n_jobs=-1,
    random_state=42,
)

clf_xgb.fit(X_train, y_train)
y_pred = clf_xgb.predict_proba(X_test)
pd.DataFrame(y_pred, columns=['predictions','isFraud']).to_csv('prediction XGBClassifier.csv')


# CatBoostClassifier without GPU

param_cb = {
        'learning_rate': 0.2,
        'bagging_temperature': 0.1, 
        'l2_leaf_reg': 30,
        'depth': 12,
        'max_bin':255,
        'iterations' : 1000,
        'loss_function' : "Logloss",
        'objective':'CrossEntropy',
        'eval_metric' : "AUC",
        'bootstrap_type' : 'Bayesian',
        'random_seed':42,
        'early_stopping_rounds' : 100,
}

clf_ctb = CatBoostClassifier(silent=True, **param_cb)

clf_ctb.fit(X_train, y_train)
y_pred = clf_ctb.predict_proba(X_test)
pd.DataFrame(y_pred, columns=['predictions','isFraud']).to_csv('prediction CatBoostClassifier.csv')


# StackNetClassifiers========

models = [  ######## First level ########
            [clf_lgb, clf_xgb, clf_ctb],
            ######## Second level ########
            [clf_lgb],
]

# StackNetClassifier with GPU

#you can convert dataframe to numpy array by .as_matrix()
X_test = X_test.as_matrix()
X_train = X_train.as_matrix()
#then refit model, it is ok

model = StackNetClassifier(
    models,
    metric="auc",
    folds=2,
    restacking=False,
    use_retraining=False,
    use_proba=True,
    random_state=42,
    verbose=1,
)

model.fit(X_train, y_train)
y_pred = model.predict_proba(X_test)
pd.DataFrame(y_pred, columns=['predictions','isFraud']).to_csv('prediction StackNetClassifier.csv')



#Neural Networks

from sklearn.neural_network import MLPClassifier

clf_nn = MLPClassifier(solver='lbfgs', activation='relu', alpha=1e-3, 
                       hidden_layer_sizes=(3), random_state = 123, verbose=False)

clf_nn.fit(X_train, y_train)
y_pred = clf_nn.predict_proba(X_test)
pd.DataFrame(y_pred, columns=['predictions','isFraud']).to_csv('prediction Neural Networks.csv')



