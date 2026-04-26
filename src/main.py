# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# %%
from pathlib import Path
import pandas as pd

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
data_dir = BASE_DIR / "data" / "raw"

train = pd.read_csv(data_dir / "train.csv")
test = pd.read_csv(data_dir / "test.csv")
# %%
train.info()
# %%
num_col = train.select_dtypes(include=[np.number]).columns
# %%
x = train.drop("SalePrice",axis = 1)
y = train["SalePrice"]
def add_features(df):
    df = df.copy()
    df["TotalSF"]= df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["TotalBath"]=(df["FullBath"] + 0.5 * df["HalfBath"] +df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"])
    df["TotalPorch"]=(df["OpenPorchSF"] + df["EnclosedPorch"] +df["3SsnPorch"]+ df["ScreenPorch"])
    df["HouseAge"]=df["YrSold"] - df["YearBuilt"]
    df["RemodAge"]=df["YrSold"] - df["YearRemodAdd"]
    df["GarageAge"]=df["YrSold"] - df["GarageYrBlt"].fillna(df["YearBuilt"])
    df["IsRemodeled"]=(df["YearBuilt"] != df["YearRemodAdd"]).astype(int)
    df["QualxSF"]=df["OverallQual"] * df["TotalSF"]
    df["OverallScore"]=df["OverallQual"] * df["OverallCond"]
    df["AreaQuality"]=df["GrLivArea"] * df["OverallQual"]
    df["AgeQuality"]=df["HouseAge"] * df["OverallQual"]
    df["TotalQuality"]=df["TotalSF"] * df["OverallQual"]
    return df

x    = add_features(x)
test = add_features(test)
# %%
num_cols = x.select_dtypes(include = [np.number]).columns
cat_col = x.select_dtypes(include=[np.object_]).columns
num_cols , cat_col
# %%
from sklearn.model_selection import train_test_split
x_train , x_val , y_train , y_val = train_test_split(x,y,test_size = 0.2 , random_state= 42)
# %%
for i in num_cols:
    inliers = x_train[i].quantile(0.99)
    x_train[i] = x_train[i].clip(upper = inliers)
    test[i] = test[i].clip(upper = inliers)
    x_val[i] = x_val[i].clip(upper = inliers)
# %%
num_df = x_train[num_cols]
num_df
# %%
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

num_pipeline = Pipeline([("imputer" , SimpleImputer(strategy='constant' , fill_value = 0)),
                        ('scaler' , StandardScaler())
                        ])
# %%
cat_df = x_train[cat_col]
cat_df
# %%
ordinal_col = ['ExterQual','ExterCond','BsmtQual','BsmtCond','BsmtExposure','HeatingQC','FireplaceQu','GarageQual','GarageCond','KitchenQual','PoolQC']

from sklearn.preprocessing import OrdinalEncoder
quality_order = ['None','Po', 'Fa', 'TA', 'Gd', 'Ex',]
categories = [quality_order for _ in range(len(ordinal_col))]
ord_pipeline = Pipeline([('imputer' , SimpleImputer(strategy="constant",fill_value="None")),
                         ('ord',OrdinalEncoder(categories =categories ,handle_unknown= "use_encoded_value" , unknown_value=-1))])
# %%
one_hot_col = ['MSZoning','Street','Alley','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition2','Condition1','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Foundation','BsmtFinType1','BsmtFinType2','Heating','CentralAir','Electrical','Functional','GarageType','GarageFinish','PavedDrive','Fence','MiscFeature','SaleType','SaleCondition']

from sklearn.preprocessing import OneHotEncoder
one_pipeline = Pipeline([('imputer' , SimpleImputer(strategy="constant",fill_value="None")),
                         ('one_hot',OneHotEncoder(handle_unknown='ignore'))])
# %%
from sklearn.compose import ColumnTransformer
full_pipe = ColumnTransformer([('num' , num_pipeline , num_cols ),
              ('cat' , one_pipeline , one_hot_col),
              ('ord' , ord_pipeline , ordinal_col)
              ])
x_train_transformed = full_pipe.fit_transform(x_train)
x_val_transformed = full_pipe.transform(x_val)
test_transformed = full_pipe.transform(test)
# %%
from sklearn.linear_model import LinearRegression
lin_model = LinearRegression()

lin_model.fit(x_train_transformed , y_train)

prediction = lin_model.predict(x_val_transformed)
# %%
from sklearn.metrics import mean_squared_error,r2_score

mse = mean_squared_error(y_val, prediction)
r2 = r2_score(y_val, prediction)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R2 Score: {r2:.4f}")
# %%
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=100,max_depth=10,random_state=42)

rf_model.fit(x_train_transformed , y_train)
rf_prediction = rf_model.predict(x_val_transformed)
# %%
from sklearn.metrics import mean_squared_error,r2_score

mse = mean_squared_error(y_val, rf_prediction)
r2 = r2_score(y_val, rf_prediction)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R2 Score: {r2:.4f}")
# %%
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


x_train_transformed = full_pipe.fit_transform(x_train)

x_val_transformed = full_pipe.transform(x_val)

xg_model = XGBRegressor(n_estimators = 1000,
learning_rate = 0.05, 
max_depth=4,           
subsample=0.8,         
colsample_bytree=0.8,  
random_state=42,
n_jobs=-1)
y_train_log = np.log1p(y_train)
xg_model.fit(x_train_transformed, y_train_log)

xg_prediction_log = xg_model.predict(x_val_transformed)
xg_prediction = np.expm1(xg_prediction_log) # نرجع السعر لأصله

# 5. التقييم (Metrics) - تأكد إنك بتبعت الـ prediction مش الموديل
mse = mean_squared_error(y_val, xg_prediction)
rmse = np.sqrt(mse)
r2 = r2_score(y_val, xg_prediction)

print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R2 Score: {r2:.4f}")
# %%
from sklearn.model_selection import cross_val_score

scores = cross_val_score(xg_model, x_train_transformed, y_train, scoring="neg_mean_squared_error", cv=5)
rmse = np.sqrt(-scores.mean())
rmse
# %%
import matplotlib.pyplot as plt

feature_names = full_pipe.get_feature_names_out()

importances = xg_model.feature_importances_

feat_imp = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print(feat_imp.head(10))
