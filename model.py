

import numpy as np
import pandas as pd
import io

pd.options.display.max_columns = None
pd.set_option('display.max_rows', 500)
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
housing = pd.read_csv("Bengaluru_House_Data.csv")
##print(housing.head())
housing_clean = housing.copy()
housing_clean.dropna(axis=0, thresh = 7, inplace = True)
housing_clean["location"] =  housing_clean["location"].replace(to_replace = np.nan, value = "Anantapura")
housing_clean["bhk"] = housing_clean["size"].apply(lambda x: int(x.split(" ")[0]))

def IsFloat(x):
    try:
        float(x)
    except:
        return False
    return True

m = []

def ConvertToSqFt(x, m):
    if m == "Acres":
        return x * 43560
    elif m == "Cents":
        return x * 435.6
    elif m == "Grounds":
        return x * 2400
    elif m == "Guntha":
        return x * 1088.98
    elif m == "Perch":
        return x * 272.25
    elif m == "Sq. Meter":
        return x * 10.7639
    elif m == "Sq. Yards":
        return x * 9
    else:
        return np.nan

def ExtractTotalSqft(x):
    try:
        values = x.split("-")
        return np.mean(list(map(float, values)))
    except ValueError:
        if x == np.nan:
            return np.nan
        else:
            for Index in range(len(x)-1, -1, -1):
                if IsFloat(x[0:Index]):
                    return ConvertToSqFt(float(x[0:Index]), x[Index:])
                
housing_clean["sqft"] = housing_clean["total_sqft"].apply(ExtractTotalSqft)     
          
def FillBathrooms(bhk_groupby_bathroom, row):
    if pd.isnull(row["bath"]):
        return int(bhk_groupby_bathroom[row["bhk"]].index[0])
    else:
        return int(row["bath"])
    
bhk_groupby_bathroom = housing_clean.groupby("bhk")["bath"].value_counts()
housing_clean["bath"] = housing_clean.apply(lambda row: FillBathrooms(bhk_groupby_bathroom, row), axis=1)

def FillBalcony(bhk_groupby_balcony, row):
    if pd.isnull(row["balcony"]):
        return int(bhk_groupby_bathroom[row["bhk"]].index[0] - 1)
    else:
        return int(row["balcony"])

bhk_groupby_balcony = housing_clean.groupby("bhk")["balcony"].value_counts()
housing_clean["balcony"] = housing_clean.apply(lambda row: FillBalcony(bhk_groupby_balcony, row), axis=1)
housing_clean.drop(["society", "size", "total_sqft"], inplace = True, axis=1)


def RelabelAvailability(x):
    values = x.split("-")
    try:
        if len(values) > 1:
            return "Soon to be Vacated"
        else:
            return x
    except:
            return ""

housing_clean["availability"] = housing_clean["availability"].apply(RelabelAvailability)
housing_clean["location"] = housing_clean["location"].apply(lambda x: x.strip())
unique_location_count = housing_clean.groupby("location")["location"].agg("count").sort_values(ascending = False)
   
unique_location_count1 = unique_location_count[unique_location_count <= 10]
housing_clean["location"] = housing_clean["location"].apply(lambda x : "Other" if x in unique_location_count1 else x)

housing_clean["price_per_sqft"] = housing_clean["price"] * 100000 / housing_clean["sqft"]
housing_clean["sqft_per_bhk"] = housing_clean["sqft"] / housing_clean["bhk"]

housing_clean = housing_clean[~(housing_clean["sqft_per_bhk"] < 300)]
housing_clean = housing_clean[~(housing_clean["sqft_per_bhk"] > 1200)]

housing_clean = housing_clean[~(housing_clean["sqft"] > 6000)]
housing_clean.sort_values(["price"], ascending=False)

housing_clean = housing_clean[~(housing_clean["price_per_sqft"] > 20000)]


def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')

housing_clean = remove_bhk_outliers(housing_clean)
housing_clean = housing_clean[housing_clean.bath < housing_clean.bhk+2]
housing_clean.drop(["price_per_sqft", "sqft_per_bhk", "balcony"], axis = 1, inplace = True)
housing_clean.to_csv("Cleaned_data.csv")
price = housing_clean["price"]
housing_clean.drop(["price"], axis = 1, inplace = True)
housing_for_model = housing_clean[["sqft", "bhk", "bath", "availability", "area_type", "location"]]
"""
le1 = LabelEncoder()
housing_availability = le1.fit_transform(housing_clean.iloc[:,3])
le2 = LabelEncoder()
housing_area_type = le2.fit_transform(housing_clean.iloc[:,4])
le3 = LabelEncoder()
housing_location = le3.fit_transform(housing_clean.iloc[:,5])
ohe1 = OneHotEncoder()
housing_availability = ohe1.fit_transform(housing_availability.reshape(-1,1))
housing_availability = pd.DataFrame(housing_availability.toarray(), columns=le1.classes_)
ohe2 = OneHotEncoder()
housing_area_type = ohe2.fit_transform(housing_area_type.reshape(-1,1))
housing_area_type = pd.DataFrame(housing_area_type.toarray(), columns=le2.classes_)
ohe3 = OneHotEncoder()
housing_location = ohe3.fit_transform(housing_location.reshape(-1,1))
housing_location = pd.DataFrame(housing_location.toarray(), columns=le3.classes_)

housing_availability.drop([housing_availability.columns[len(housing_availability.columns)-1]], axis=1, inplace = True)
housing_area_type.drop([housing_area_type.columns[len(housing_area_type.columns)-1]], axis=1, inplace = True)
housing_location.drop([housing_location.columns[len(housing_location.columns)-1]], axis=1, inplace = True)
housing_num_features = housing_clean.iloc[:, 0:3].reset_index()
housing_num_features.drop(["index"], axis = 1, inplace = True)

std_scaler = StandardScaler()
housing_num_scaled_features = pd.DataFrame(std_scaler.fit_transform(housing_num_features), columns=housing_num_features.columns)
housing_for_model = pd.concat([housing_num_scaled_features, housing_availability, housing_area_type, housing_location], axis=1)
"""

X, y = housing_for_model.values, price.values

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error


X_train, X_test, y_train, y_test = train_test_split(housing_for_model, price, test_size = 0.2, random_state = 0)

col_trans=make_column_transformer((OneHotEncoder(sparse_output=False),['location','area_type','availability']),remainder='passthrough')

scaler=StandardScaler()
"""
def find_best_model_using_gridsearchcv(X, y):
    algos = {
        'linear_regression' : {
            'model': LinearRegression(),
            'params': {
                'normalize': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [0.1, 0.3, 0.5, 0.7, 0.9],
                'selection': ['random', 'cyclic']
            }
        },
        'ridge': {
            'model': Ridge(),
            'params': {
                'alpha': [0.1, 0.3, 0.5, 0.7, 0.9]
            }
        },
        'random_forest': {
            'model': RandomForestRegressor(),
            'params': {
                'n_estimators': [10, 20, 500], 'max_depth': [2, 4, 6, 8], 
            }
        },
        'xgboost': {
            'model': XGBRegressor(),
            'params': {
                'n_estimators': [100, 200, 300], 
            }
        }

    }

    
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores,columns=['model','best_score','best_params'])

pddata=find_best_model_using_gridsearchcv(X_train, y_train)
#df = pd.DataFrame(pddata)
print(pddata)
"""
lin_reg = LinearRegression()
pipe=make_pipeline(col_trans,scaler,lin_reg)
pipe.fit(X_train, y_train)
y_p1=pipe.predict(X_test)
print(r2_score(y_test,y_p1))

ridge_reg = Ridge(alpha = 0.1)
pipe=make_pipeline(col_trans,scaler,ridge_reg)
pipe.fit(X_train, y_train)
y_p2=pipe.predict(X_test)
print(r2_score(y_test,y_p2))

lasso_reg = Lasso(alpha = 0.1)
pipe=make_pipeline(col_trans,scaler,lasso_reg)
pipe.fit(X_train, y_train)
y_p3=pipe.predict(X_test)
print(r2_score(y_test,y_p3))

dt_reg = DecisionTreeRegressor()
pipe=make_pipeline(col_trans,scaler,dt_reg)
pipe.fit(X_train, y_train)
y_p4=pipe.predict(X_test)
print(r2_score(y_test,y_p4))

rf_reg = RandomForestRegressor()
pipe=make_pipeline(col_trans,scaler,rf_reg)
pipe.fit(X_train, y_train)
y_p5=pipe.predict(X_test)
print(r2_score(y_test,y_p5))

xgb_reg = XGBRegressor()
pipe=make_pipeline(col_trans,scaler,xgb_reg)
pipe.fit(X_train, y_train)
y_p6=pipe.predict(X_test)
print(r2_score(y_test,y_p6))
"""
def find_best_model_using_gridsearchcv(X, y):
    algos = {
        'linear_regression' : {
            'model': LinearRegression(),
            'params': {
                'normalize': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [0.1, 0.3, 0.5, 0.7, 0.9],
                'selection': ['random', 'cyclic']
            }
        },
        'ridge': {
            'model': Ridge(),
            'params': {
                'alpha': [0.1, 0.3, 0.5, 0.7, 0.9]
            }
        },
        'random_forest': {
            'model': RandomForestRegressor(),
            'params': {
                'n_estimators': [10, 20, 500], 'max_depth': [2, 4, 6, 8], 
            }
        },
        'xgboost': {
            'model': XGBRegressor(),
            'params': {
                'n_estimators': [100, 200, 300], 
            }
        }

    }

    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores,columns=['model','best_score','best_params'])

xgb_reg = XGBRegressor()
pipe=make_pipeline(col_trans,scaler,xgb_reg)
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))

"""

import pickle
pickle.dump(pipe,open('model.pkl','wb'))