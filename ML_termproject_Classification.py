import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

VEHICLE = pd.read_csv('datasets/VEHICLE.csv',
                      usecols=['ST_CASE', 'VEH_NO', 'TRAV_SP', 'DR_DRINK', 'PREV_ACC', 'VSPD_LIM', 'VALIGN', 'VPROFILE',
                               'VPAVETYP', 'VSURCOND', 'VTRAFCON'])
print(VEHICLE.ST_CASE.value_counts())

# Set NaN
VEHICLE['TRAV_SP'].replace([997, 998, 999], np.NaN, inplace=True)  # if greater than 151 -> 997
VEHICLE['PREV_ACC'].replace([98, 99, 998], np.NaN, inplace=True)
VEHICLE['VSPD_LIM'].replace([98, 99], np.NaN, inplace=True)
VEHICLE['VALIGN'].replace([8, 9], np.NaN, inplace=True)
VEHICLE['VPROFILE'].replace([8, 9], np.NaN, inplace=True)
VEHICLE['VPAVETYP'].replace([8, 9], np.NaN, inplace=True)
VEHICLE['VSURCOND'].replace([98, 99], np.NaN, inplace=True)
VEHICLE['VTRAFCON'].replace([97, 99], np.NaN, inplace=True)

ACCIDENT = pd.read_csv('datasets/ACCIDENT.csv',
                       usecols=['ST_CASE', 'HOUR', 'ROUTE', 'TYP_INT', 'LGT_COND', 'WEATHER', 'FATALS'])
# Set NaN
ACCIDENT['HOUR'].replace(99, np.NaN, inplace=True)
ACCIDENT['ROUTE'].replace(9, np.NaN, inplace=True)
ACCIDENT['TYP_INT'].replace([98, 99], np.NaN, inplace=True)
ACCIDENT['WEATHER'].replace([98, 99], np.NaN, inplace=True)
ACCIDENT['LGT_COND'].replace(9, np.NaN, inplace=True)

DISTRACT = pd.read_csv('datasets/DISTRACT.csv', usecols=['ST_CASE', 'VEH_NO', 'MDRDSTRD'])
DISTRACT.replace(99, np.NaN, inplace=True)  # Set NaN (value: 99)

VISION = pd.read_csv('datasets/VISION.csv', usecols=['ST_CASE', 'VEH_NO', 'MVISOBSC'])
VISION.replace(99, np.NaN, inplace=True)  # Set NaN (value: 99)

df = pd.merge(VEHICLE, ACCIDENT, left_on='ST_CASE', right_on='ST_CASE', how='left')
df = pd.merge(df, DISTRACT, left_on=['ST_CASE', 'VEH_NO'], right_on=['ST_CASE', 'VEH_NO'], how='inner')
df = pd.merge(df, VISION, left_on=['ST_CASE', 'VEH_NO'], right_on=['ST_CASE', 'VEH_NO'], how='left')

df.drop(['TRAV_SP'], axis=1, inplace=True)
df.dropna(inplace=True)

df.loc[df['FATALS'].values == 1, 'FATALS'] = 0
df.loc[df['FATALS'].values > 1, 'FATALS'] = 1

x = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]
columns_type = dict(zip(df.columns, x))

for key, value in columns_type.items():
    if value == 1:
        df[key] = LabelEncoder().fit_transform(df[key])
df.info()

X_train, X_valid, y_train, y_valid = \
    train_test_split(df.drop(['FATALS'], axis=1), df['FATALS'], test_size=0.2, stratify=df['FATALS'])

models = list()

models.append(LogisticRegression())
models.append(DecisionTreeClassifier())
models.append(RandomForestClassifier())
models.append(XGBClassifier())

print(df['FATALS'].value_counts())

param_grid = [{'C': [0.001, 0.01, 0.1, 1, 10, 100], 'max_iter': [0.1, 1, 10, 100]},
              {'criterion': ['gini', 'entropy'], 'max_depth': [3, 5, 7, 10]},
              {'n_estimators': [0.1, 1, 10, 100], 'criterion': ['gini', 'entropy']},
              {'booster': ['gbtree', 'gblinear', 'dart']}]

for i in range(len(models)):
    grid_search = GridSearchCV(models[i], param_grid[i], cv=5, return_train_score=True)
    grid_search.fit(X_train, y_train)

    print("Best Accuracy: {:.2f}".format(grid_search.score(X_valid, y_valid)))
    print("Best Parameters:", grid_search.best_params_)
    print("Best Model: {:.2f}".format(grid_search.best_score_))

    print(confusion_matrix(grid_search.predict(X_valid), y_valid))

    sns.heatmap(confusion_matrix(grid_search.predict(X_valid), y_valid), annot=True, fmt='d')

    plt.show()

