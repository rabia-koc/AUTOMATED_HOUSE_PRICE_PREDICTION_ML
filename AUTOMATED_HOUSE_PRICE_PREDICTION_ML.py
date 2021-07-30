import warnings

import numpy as np
import pandas as pd
import joblib
import argparse

import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import  RandomizedSearchCV
import numpy as np
from scipy.stats import norm

from scipy import stats
from scipy.stats import skew
from scipy.stats import skew, boxcox_normmax, norm
from scipy.special import boxcox1p

import time
from contextlib import contextmanager

# from helpers.data_prep import *
# from helpers.eda import *

warnings.simplefilter(action='ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from warnings import filterwarnings
filterwarnings('ignore')


# decaratorlör: fonksiyon biçimlendiren fonksiyonlardır.
@contextmanager
# zaman bilgisini yansıtmak için
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))
    print(" ")


def get_namespace():
    parser = argparse.ArgumentParser()

    parser.add_argument('--nobase', dest='base', action='store_false')
    parser.set_defaults(base=True)

    parser.add_argument('--dump', dest='dump', action='store_true')
    parser.set_defaults(dump=False)

    parser.add_argument('--scoring', dest="scoring", action="store", type=str)
    parser.set_defaults(scoring="neg_mean_squared_error")

    return parser.parse_args()


######################################################
# Data Preprocessing & Feature Engineering
######################################################
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # print(f"Observations: {dataframe.shape[0]}")
    # print(f"Variables: {dataframe.shape[1]}")
    # print(f'cat_cols: {len(cat_cols)}')
    # print(f'num_cols: {len(num_cols)}')
    # print(f'cat_but_car: {len(cat_but_car)}')
    # print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


def find_correlation(dataframe, numeric_cols, corr_limit=0.50):
    high_correlations = []
    low_correlations = []
    for col in numeric_cols:
        if col == "SalePrice":
            pass
        else:
            correlation = dataframe[[col, "SalePrice"]].corr().loc[col, "SalePrice"]
            print(col, correlation)
            if abs(correlation) > corr_limit:
                high_correlations.append(col + ": " + str(correlation))
            else:
                low_correlations.append(col + ": " + str(correlation))
    return low_correlations, high_correlations


def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list


def outliers(dataframe, num_cols):
    outlier_indices = []
    for feature in num_cols:
        print('DEĞİŞKEN: {}'.format(feature))
        q1 = np.percentile(dataframe[feature], 5)
        q3 = np.percentile(dataframe[feature], 95)
        iqr = q3 - q1
        outlier_step = iqr * 1.5
        upper = q3 + outlier_step
        lower = q1 - outlier_step
        outlier_list_col = dataframe[(dataframe[feature] < lower) | (dataframe[feature] > upper)].index
        print('AYKIRI DEĞER SAYISI: {}'.format(outlier_list_col.shape[0]), '\n')
        for a in outlier_list_col:
            outlier_indices.append(a)


def house_price_prep(dataframe):
    print("Data Preprocessing...")
    check_df(dataframe)

    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)
    outliers(dataframe, num_cols)
    # Veri kümesinin yazarı, '4000 fit kareden fazla olan evlerin' veri kümesinden çıkarılmasını önerir.
    dataframe = dataframe.drop(dataframe[(dataframe['GrLivArea'] > 4000) & (dataframe['SalePrice'] < 300000)].index)

    # MISSING VALUES
    missing_vs_target(dataframe, "SalePrice", missing_values_table(dataframe, na_name=True))
    missing_values_table(dataframe)

    dataframe["Alley"] = dataframe["Alley"].fillna("None")
    dataframe["PoolQC"] = dataframe["PoolQC"].fillna("None")
    dataframe["MiscFeature"] = dataframe["MiscFeature"].fillna("None")
    dataframe["Fence"] = dataframe["Fence"].fillna("None")
    dataframe["FireplaceQu"] = dataframe["FireplaceQu"].fillna("None")
    dataframe["LotFrontage"] = dataframe.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

    for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
        dataframe[col] = dataframe[col].fillna('None')

    dataframe.drop(['GarageArea'], axis=1, inplace=True)
    dataframe.drop(['GarageYrBlt'], axis=1, inplace=True)
    dataframe.drop(['Utilities'], axis=1, inplace=True)

    dataframe["GarageCars"] = dataframe["GarageCars"].fillna(0)

    for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
        dataframe[col] = dataframe[col].fillna(0)

    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
        dataframe[col] = dataframe[col].fillna('None')

    dataframe['MSZoning'] = dataframe.groupby('MSSubClass')['MSZoning'].apply(lambda x: x.fillna(x.mode()[0]))
    dataframe["MasVnrType"] = dataframe["MasVnrType"].fillna("None")
    dataframe["MasVnrArea"] = dataframe["MasVnrArea"].fillna(0)

    dataframe["Functional"] = dataframe["Functional"].fillna("Typ")
    dataframe['Electrical'] = dataframe['Electrical'].fillna(dataframe['Electrical'].mode()[0])
    dataframe['KitchenQual'] = dataframe['KitchenQual'].fillna(dataframe['KitchenQual'].mode()[0])
    dataframe['Exterior1st'] = dataframe['Exterior1st'].fillna(dataframe['Exterior1st'].mode()[0])
    dataframe['Exterior2nd'] = dataframe['Exterior2nd'].fillna(dataframe['Exterior2nd'].mode()[0])
    dataframe['SaleType'] = dataframe['SaleType'].fillna(dataframe['SaleType'].mode()[0])

    dataframe['YrSold'] = dataframe['YrSold'].astype(str)
    dataframe.loc[2590, 'GarageYrBlt'] = 2007

    # FEATURE ENGINEERING
    # Kategorik Değişkenler İçin
    # Öncelikle bazı kategorik değişkenleri sınıflarının bağımlı değişkene olan etkisine bakarak gruplayacağız.

    dataframe.groupby("Neighborhood").agg({"SalePrice": "mean"}).sort_values(by="SalePrice", ascending=False)  # Fiyata en büyük etkisi olan mahalle NoRidge iken, en küçük etkisi olan mahalle MeadowV'dir. Bu ortalama değerlere göre bir sıralı gruplama yapacağız.

    nhood_map = {'MeadowV': 1, 'IDOTRR': 1, 'BrDale': 1,
                 'BrkSide': 2, 'Edwards': 2, 'OldTown': 2,
                 'Sawyer': 3, 'Blueste': 3,
                 'SWISU': 4, 'NPkVill': 4, 'NAmes': 4, 'Mitchel': 4,
                 'SawyerW': 5, 'NWAmes': 5,
                 'Gilbert': 6, 'Blmngtn': 6, 'CollgCr': 6,
                 'Crawfor': 7, 'ClearCr': 7,
                 'Somerst': 8, 'Veenker': 8, 'Timber': 8,
                 'StoneBr': 9, 'NridgHt': 9,
                 'NoRidge': 10}

    dataframe['Neighborhood'] = dataframe['Neighborhood'].map(nhood_map).astype('int')

    dataframe = dataframe.replace({"MSSubClass": {20: "SC20", 30: "SC30", 40: "SC40", 45: "SC45", \
                                                  50: "SC50", 60: "SC60", 70: "SC70", 75: "SC75", \
                                                  80: "SC80", 85: "SC85", 90: "SC90", 120: "SC120", \
                                                  150: "SC150", 160: "SC160", 180: "SC180", 190: "SC190"},
                                   "MoSold": {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun", \
                                              7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
                                   })

    func = {"Sal": 0, "Sev": 1, "Maj2": 2, "Maj1": 3, "Mod": 4, "Min2": 5, "Min1": 6, "Typ": 7}
    dataframe["Functional"] = dataframe["Functional"].map(func).astype("int")
    dataframe.groupby("Functional").agg({"SalePrice": "mean"})
    # Belli bir dereceyi ifade eden sınıfları olan değişkenleri ordinal yapıya getirme.
    # MSZoning
    dataframe.loc[(dataframe["MSZoning"] == "C (all)"), "MSZoning"] = 1
    dataframe.loc[(dataframe["MSZoning"] == "RM"), "MSZoning"] = 2
    dataframe.loc[(dataframe["MSZoning"] == "RH"), "MSZoning"] = 2
    dataframe.loc[(dataframe["MSZoning"] == "RL"), "MSZoning"] = 3
    dataframe.loc[(dataframe["MSZoning"] == "FV"), "MSZoning"] = 3
    # LotShape
    dataframe.groupby("LotShape").agg({"SalePrice": "mean"}).sort_values(by="SalePrice", ascending=False)
    shape_map = {"Reg": 1, "IR1": 2, "IR3": 3, "IR2": 4}
    dataframe['LotShape'] = dataframe['LotShape'].map(shape_map).astype('int')
    # LandContour
    dataframe.groupby("LandContour").agg({"SalePrice": "mean"}).sort_values(by="SalePrice", ascending=False)
    contour_map = {"Bnk": 1, "Lvl": 2, "Low": 3, "HLS": 4}
    dataframe['LandContour'] = dataframe['LandContour'].map(contour_map).astype('int')

    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)
    rare_analyser(dataframe, "SalePrice", cat_cols)

    # LotConfig
    dataframe.loc[(dataframe["LotConfig"] == "Inside"), "LotConfig"] = 1
    dataframe.loc[(dataframe["LotConfig"] == "FR2"), "LotConfig"] = 1
    dataframe.loc[(dataframe["LotConfig"] == "Corner"), "LotConfig"] = 1
    dataframe.loc[(dataframe["LotConfig"] == "FR3"), "LotConfig"] = 2
    dataframe.loc[(dataframe["LotConfig"] == "CulDSac"), "LotConfig"] = 2

    # Condition1
    cond1_map = {"Artery": 1, "RRAe": 1, "Feedr": 1,
                 "Norm": 2, "RRAn": 2, "RRNe": 2,
                 "PosN": 3, "RRNn": 3, "PosA": 3}
    dataframe['Condition1'] = dataframe['Condition1'].map(cond1_map).astype('int')
    #dataframe['Condition1'].isnull().sum()

    # BldgType
    dataframe.loc[(dataframe["BldgType"] == "2fmCon"), "BldgType"] = 1
    dataframe.loc[(dataframe["BldgType"] == "Duplex"), "BldgType"] = 1
    dataframe.loc[(dataframe["BldgType"] == "Twnhs"), "BldgType"] = 1
    dataframe.loc[(dataframe["BldgType"] == "1Fam"), "BldgType"] = 2
    dataframe.loc[(dataframe["BldgType"] == "TwnhsE"), "BldgType"] = 2

    # RoofStyle
    dataframe.groupby("RoofStyle").agg({"SalePrice": "mean"}).sort_values(by="SalePrice", ascending=False)
    dataframe.loc[(dataframe["RoofStyle"] == "Gambrel"), "RoofStyle"] = 1
    dataframe.loc[(dataframe["RoofStyle"] == "Gablee"), "RoofStyle"] = 2
    dataframe.loc[(dataframe["RoofStyle"] == "Mansard"), "RoofStyle"] = 3
    dataframe.loc[(dataframe["RoofStyle"] == "Flat"), "RoofStyle"] = 4
    dataframe.loc[(dataframe["RoofStyle"] == "Hip"), "RoofStyle"] = 5
    dataframe.loc[(dataframe["RoofStyle"] == "Shed"), "RoofStyle"] = 6

    # RoofMatl
    dataframe.groupby("RoofMatl").agg({"SalePrice": "mean"}).sort_values(by="SalePrice", ascending=False)
    dataframe.loc[(dataframe["RoofMatl"] == "Roll"), "RoofMatl"] = 1
    dataframe.loc[(dataframe["RoofMatl"] == "ClyTile"), "RoofMatl"] = 2
    dataframe.loc[(dataframe["RoofMatl"] == "CompShg"), "RoofMatl"] = 3
    dataframe.loc[(dataframe["RoofMatl"] == "Metal"), "RoofMatl"] = 3
    dataframe.loc[(dataframe["RoofMatl"] == "Tar&Grv"), "RoofMatl"] = 3
    dataframe.loc[(dataframe["RoofMatl"] == "WdShake"), "RoofMatl"] = 4
    dataframe.loc[(dataframe["RoofMatl"] == "Membran"), "RoofMatl"] = 4
    dataframe.loc[(dataframe["RoofMatl"] == "WdShngl"), "RoofMatl"] = 5

    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)
    rare_analyser(dataframe, "SalePrice", cat_cols)

    # ExterQual
    dataframe.groupby("ExterQual").agg({"SalePrice": "mean"}).sort_values(by="SalePrice", ascending=False)
    ext_map = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
    dataframe['ExterQual'] = dataframe['ExterQual'].map(ext_map).astype('int')

    # ExterCond
    ext_map = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
    dataframe['ExterCond'] = dataframe['ExterCond'].map(ext_map).astype('int')

    # BsmtQual
    bsm_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
    dataframe['BsmtQual'] = dataframe['BsmtQual'].map(bsm_map).astype('int')

    # BsmtCond
    bsm_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
    dataframe['BsmtCond'] = dataframe['BsmtCond'].map(bsm_map).astype('int')

    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)
    rare_analyser(dataframe, "SalePrice", cat_cols)

    # BsmtFinType1
    bsm_map = {'None': 0, 'Rec': 1, 'BLQ': 1, 'LwQ': 2, 'ALQ': 3, 'Unf': 3, 'GLQ': 4}
    dataframe['BsmtFinType1'] = dataframe['BsmtFinType1'].map(bsm_map).astype('int')

    # BsmtFinType2
    bsm_map = {'None': 0, 'BLQ': 1, 'Rec': 2, 'LwQ': 2, 'Unf': 3, 'GLQ': 3, 'ALQ': 4}
    dataframe['BsmtFinType2'] = dataframe['BsmtFinType2'].map(bsm_map).astype('int')

    # BsmtExposure
    bsm_map = {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}
    dataframe['BsmtExposure'] = dataframe['BsmtExposure'].map(bsm_map).astype('int')

    # Heating
    heat_map = {'Floor': 1, 'Grav': 1, 'Wall': 2, 'OthW': 3, 'GasW': 4, 'GasA': 5}
    dataframe['Heating'] = dataframe['Heating'].map(heat_map).astype('int')

    # HeatingQC
    heat_map = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
    dataframe['HeatingQC'] = dataframe['HeatingQC'].map(heat_map).astype('int')

    # KitchenQual
    kitch_map = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
    dataframe['KitchenQual'] = dataframe['KitchenQual'].map(heat_map).astype('int')

    # FireplaceQu
    fire_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
    dataframe['FireplaceQu'] = dataframe['FireplaceQu'].map(fire_map).astype('int')

    # GarageCond
    garage_map = {'None': 1, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
    dataframe['GarageCond'] = dataframe['GarageCond'].map(garage_map).astype('int')
    #dataframe['GarageCond'].value_counts()

    # GarageQual
    garage_map = {'None': 1, 'Po': 1, 'Fa': 2, 'TA': 3, 'Ex': 4, 'Gd': 5}
    dataframe['GarageQual'] = dataframe['GarageQual'].map(garage_map).astype('int')

    # PavedDrive
    paved_map = {'N': 1, 'P': 2, 'Y': 3}
    dataframe['PavedDrive'] = dataframe['PavedDrive'].map(paved_map).astype('int')

    # CentralAir: Merkezi klima
    cent = {"N": 0, "Y": 1}
    dataframe["CentralAir"] = dataframe["CentralAir"].map(cent).astype("int")
    dataframe.groupby("CentralAir").agg({"SalePrice": "mean"})

    # LandSlope
    dataframe.loc[dataframe["LandSlope"] == "Gtl", "LandSlope"] = 1
    dataframe.loc[dataframe["LandSlope"] == "Sev", "LandSlope"] = 2
    dataframe.loc[dataframe["LandSlope"] == "Mod", "LandSlope"] = 2
    dataframe["LandSlope"] = dataframe["LandSlope"].astype("int")

    # OverallQual
    dataframe.loc[dataframe["OverallQual"] == 1, "OverallQual"] = 1
    dataframe.loc[dataframe["OverallQual"] == 2, "OverallQual"] = 1
    dataframe.loc[dataframe["OverallQual"] == 3, "OverallQual"] = 1
    dataframe.loc[dataframe["OverallQual"] == 4, "OverallQual"] = 2
    dataframe.loc[dataframe["OverallQual"] == 5, "OverallQual"] = 3
    dataframe.loc[dataframe["OverallQual"] == 6, "OverallQual"] = 4
    dataframe.loc[dataframe["OverallQual"] == 7, "OverallQual"] = 5
    dataframe.loc[dataframe["OverallQual"] == 8, "OverallQual"] = 6
    dataframe.loc[dataframe["OverallQual"] == 9, "OverallQual"] = 7
    dataframe.loc[dataframe["OverallQual"] == 10, "OverallQual"] = 8

    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)
    rare_analyser(dataframe, "SalePrice", cat_cols)

    #################################################

    # FEATURE ENGINEERING FOR OTHER FEATURES
    # Evin genel durumunu
    dataframe["OverallGrade"] = dataframe["OverallQual"] * dataframe["OverallCond"]
    # garaj kalitesi ile garaj durumu
    dataframe["GarageGrade"] = dataframe["GarageQual"] * dataframe["GarageCond"]
    # şömine kalitesi için
    dataframe["FireplaceScore"] = dataframe["Fireplaces"] * dataframe["FireplaceQu"]
    # Ev için toplam SF (bodrum dahil)
    dataframe["AllSF"] = dataframe["GrLivArea"] + dataframe["TotalBsmtSF"]
    # 1. + 2. katlar için toplam SF
    dataframe["AllFlrsSF"] = dataframe["1stFlrSF"] + dataframe["2ndFlrSF"]

    dataframe['TotalSF'] = (dataframe['BsmtFinSF1'] +
                            dataframe['BsmtFinSF2'] +
                            dataframe['1stFlrSF'] +
                            dataframe['2ndFlrSF'])
    # toplam banyolar
    dataframe['TotalBathrooms'] = (dataframe['FullBath'] +
                                   (0.5 * dataframe['HalfBath']) +
                                   dataframe['BsmtFullBath'] +
                                   (0.5 * dataframe['BsmtHalfBath']))

    # toplam sundurma alanları
    dataframe['TotalPorchSF'] = (dataframe['OpenPorchSF'] +
                                 dataframe['3SsnPorch'] +
                                 dataframe['EnclosedPorch'] +
                                 dataframe['ScreenPorch'])

    # Kategorilern kalitelerini kendi içlerinde toplayalım:
    # dış cephedeki malzemenin mevcut durumunu ve kalitesini toplayarak dış cephe malzemesi kalitesini bulma.
    dataframe['TotalExtQual'] = (dataframe['ExterQual'] +
                                 dataframe['ExterCond'])

    # Bodrum kalitesi için
    dataframe['TotalBsmQual'] = (dataframe['BsmtQual'] +
                                 dataframe['BsmtCond'] +
                                 dataframe['BsmtFinType1'] +
                                 dataframe['BsmtFinType2'])

    # Tüm kaliteleri toplayalım:
    dataframe['TotalQual'] = dataframe['OverallQual'] + \
                             dataframe['TotalExtQual'] + \
                             dataframe['TotalBsmQual'] + \
                             dataframe['KitchenQual'] + \
                             dataframe['HeatingQC']

    dataframe['HasPool'] = dataframe['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
    dataframe['Has2ndFloor'] = dataframe['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
    dataframe['hasgarage'] = dataframe['GarageCars'].apply(lambda x: 1 if x > 0 else 0)
    dataframe['HasFireplace'] = dataframe['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
    dataframe['HasPorch'] = dataframe['TotalPorchSF'].apply(lambda x: 1 if x > 0 else 0)
    dataframe['HasBsmt'] = dataframe['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
    dataframe["hascentralAir"] = dataframe["CentralAir"].apply(lambda x: 1 if x > 0 else 0)

    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)

    corr = dataframe.corr()
    corr.sort_values(["SalePrice"], ascending=False, inplace=True)
    print(corr.SalePrice[:11])

    low_corrs, high_corrs = find_correlation(dataframe, num_cols)
    high_correlated_cols(dataframe, False, 0.80)

    # RARE ENCODING
    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)
    rare_analyser(dataframe, "SalePrice", cat_cols)

    dataframe = rare_encoder(dataframe, 0.01, cat_cols)
    rare_analyser(dataframe, "SalePrice", cat_cols)

    useless_cols = [col for col in cat_cols if dataframe[col].nunique() == 1 or
                    (dataframe[col].nunique() == 2 and (dataframe[col].value_counts() / len(dataframe) <= 0.01).any(axis=None))]
    dataframe[useless_cols].head()

    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)

    # cat_cols güncellemesi
    cat_cols = [col for col in cat_cols if col not in useless_cols]

    # gereksiz kolonları dataframeden silinmesi
    for col in useless_cols:
        dataframe.drop(col, axis=1, inplace=True)

    rare_analyser(dataframe, "SalePrice", cat_cols)
    #dataframe.shape

    ##################
    # Label Encoding & One-Hot Encoding
    ##################
    cat_cols = cat_cols + cat_but_car
    dataframe = one_hot_encoder(dataframe, cat_cols, drop_first=True)
    check_df(dataframe)
    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)
    rare_analyser(dataframe, "SalePrice", cat_cols)
    useless_cols_new = [col for col in cat_cols if (dataframe[col].value_counts() / len(dataframe) <= 0.01).any(axis=None)]

    for col in useless_cols_new:
        cat_summary(dataframe, col)

    for col in useless_cols_new:
        dataframe.drop(col, axis=1, inplace=True)

    #dataframe.shape
    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)
    rare_analyser(dataframe, "SalePrice", cat_cols)
    return dataframe

######################################################
# Base Models
######################################################
def base_models(x, y, scoring):
    print("Base Models...")
    models = [('LR', LinearRegression()),
              ("Ridge", Ridge()),
              ("Lasso", Lasso()),
              ("ElasticNet", ElasticNet()),
              ('KNN', KNeighborsRegressor()),
              ('CART', DecisionTreeRegressor()),
              ('RF', RandomForestRegressor()),
              ('SVR', SVR()),
              ('GBM', GradientBoostingRegressor()),
              ("XGBoost", XGBRegressor(objective='reg:squarederror')),
              ("LightGBM", LGBMRegressor()),
              ("CatBoost", CatBoostRegressor(verbose=False))]

    for name, regressor in models:
        rmse = np.mean(np.sqrt(-cross_val_score(regressor, x, y, cv=5, scoring="neg_mean_squared_error")))
        print(f"RMSE: {round(rmse, 4)} ({name}) ")
# RMSE: 0.123 (LightGBM)

######################################################
# Automated Hyperparameter Optimization
######################################################
def hyperparameter_optimization(x, y, cv=10, scoring="neg_mean_squared_error"):
    print("Hyperparameter Optimization....")

    # lightgbm_params = {'boosting_type': ['gbdt', 'dart', 'goss'],
    #                    "learning_rate": [0.01, 0.03, 0.05, 0.07, 0.1, 0.2],
    #                    "n_estimators": [100, 150, 200, 300, 400, 500],
    #                    "max_depth": [3, 5, 8],
    #                    "colsample_bytree": [0.3, 0.4, 0.5, 0.7, 0.8, 1]}

    best_params= {'boosting_type': 'gbdt',
                  'colsample_bytree': 0.3,
                  'learning_rate': 0.03,
                  'max_depth': 3,
                  'n_estimators': 500}

    models = [('LightGBM', LGBMRegressor(random_state=46), best_params)]

    for name, regressor, params in models:
        print(f"########## {name} ##########")
        rmse = np.mean(np.sqrt(-cross_val_score(regressor, x, y, cv=5, scoring=scoring)))
        print(f"RMSE BEFORE: {round(rmse, 4)} ({name}) ")
        final_model = regressor.set_params(**best_params).fit(x,y)
        results_rmse = np.mean(np.sqrt(-cross_val_score(final_model, x, y, cv=cv, scoring=scoring)))
        print(f"RMSE AFTER: {round(results_rmse, 4)} ({name}) ")
        # 0.11413422992471374

        # Feature Selection
        # i = 1
        # while i < 8:
        #     zero_imp_cols = feature_imp[feature_imp["Value"] < i]["Feature"].values
        #     selected_cols = [col for col in x.columns if col not in zero_imp_cols]
        #     i = i + 0.1
        #     final_model = lgbm_model.set_params(**lgbm_gs_best.best_params_).fit(x[selected_cols], y)
        #     rmse = np.mean(
        #         np.sqrt(-cross_val_score(final_model, x[selected_cols], y, cv=5, scoring="neg_mean_squared_error")))
        #     print(i)
        #     print(len(selected_cols))
        #     print(rmse)

        feature_imp = pd.DataFrame({'Value': final_model.feature_importances_, 'Feature': x.columns})
        zero_imp_cols = feature_imp[feature_imp["Value"] < 6.2]["Feature"].values
        selected_cols = [col for col in x.columns if col not in zero_imp_cols]

        print("Hyperparameter Optimization with Selected Features...")
        print(f"########## {name} ##########")
        rmse = np.mean(np.sqrt(-cross_val_score(regressor, x[selected_cols], y, cv=5, scoring=scoring)))
        print(f"RMSE BEFORE: {round(rmse, 4)} ({name}) ")
        final_model = regressor.set_params(**best_params).fit(x[selected_cols],y)
        results_rmse = np.mean(np.sqrt(-cross_val_score(final_model, x[selected_cols], y, cv=cv, scoring=scoring)))
        print(f"RMSE AFTER: {round(results_rmse, 4)} ({name}) ")
        # 0.11297465470162453

    return final_model, selected_cols


#######################################
# Sonuçların Yüklenmesi
#######################################
def Predict(final_model, selected_cols, test_df):
    print("Predict...")
    submission_df = pd.DataFrame()
    submission_df['Id'] = test_df["Id"]  # kaggle'nin bana verdiği id'leri
    y_pred_sub = final_model.predict(test_df[selected_cols])
    # gerçek tahmin edilen değerler yani gerçek tahmin edilen hatalar elimizde şatış fiyatları.
    y_pred_sub = np.expm1(y_pred_sub)  # logaritmayı geri aldık.
    submission_df['SalePrice'] = y_pred_sub
    return submission_df

######################################################
# Main
######################################################

def main(base, dump, scoring):

    with timer("Data Preprocessing"):
        train = pd.read_csv("..\house_prices\train.csv")
        test = pd.read_csv("..\house_prices\test.csv")
        df = train.append(test).reset_index(drop=True)
        df_ = house_price_prep(dataframe=df)

        test_df = df_[df_['SalePrice'].isnull()].drop("SalePrice", axis=1)
        train_df = df_[df_['SalePrice'].notnull()]

        y = np.log1p(train_df['SalePrice'])
        x = train_df.drop(["Id", "SalePrice"], axis=1)

    if base:
        with timer("Base Models"):
            base_models(x, y, scoring)

    with timer("Hyperparameter Optimization"):
        final_model, selected_cols = hyperparameter_optimization(x, y, cv=10, scoring=scoring)

    with timer("Predict"):

        submission_df = Predict(final_model, selected_cols, test_df)
        if dump:
            print("Predict Model Saved")
            submission_df.to_csv('submission.csv', index=False)
            joblib.dump(submission_df, "predict_clf.pkl")


if __name__ == "__main__":

    namespace = get_namespace()

    with timer("Full Script Running Time"):
        main(base=namespace.base, dump=namespace.dump, scoring=namespace.scoring)


