import os
os.chdir('./src')
from dataloader import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
from sklearn.linear_model import ElasticNet
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

import pickle
import joblib

import warnings
warnings.filterwarnings('ignore')

# os.chdir('./src')

# 피해액 책정 모델 구축
def price_prediction_models(ames):
    """라쏘 및 릿지 회귀를 활용한 피해액 예측 모델 구축"""
    
    dataloader = DataLoader()
    ames = dataloader.load_data()
    
    target = 'SalePrice'
    y = ames[target]
    
    ig_cols = ['PID','SalePrice','GeoRefNo','Prop_Addr','Latitude','Longitude','Date',
               'Risk_RoofMatl','Risk_Exterior1st','Risk_Exterior2nd','Risk_MasVnrType','Risk_WoodDeckSF',
               'BuildingPricePerTotalSF','BuildingValue', 'LandValue', 'TotalSF']
    X = ames.drop(columns=ig_cols)
    
    # 학습/테스트 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    num_columns = X_train.select_dtypes('number').columns.tolist()
    cat_columns = X_train.select_dtypes('object').columns.tolist()

    cat_preprocess = make_pipeline(
    OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    )

    num_preprocess = make_pipeline(SimpleImputer(strategy="mean"), 
                                StandardScaler())

    preprocessor = ColumnTransformer(
    [("num", num_preprocess, num_columns),
    ("cat", cat_preprocess, cat_columns)]
    )
    
    # # Elastic 모델
    elastic_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', ElasticNet(max_iter=10000))
    ])
    
    # 알파값 설정
    elastic_params = {
        'regressor__alpha':np.arange(0.1, 1, 0.1),
        'regressor__l1_ratio': np.linspace(0,1,5)
    }
    
    # K-폴드 교차 검증
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Elastic 모델 튜닝
    elastic_grid = GridSearchCV(
        elastic_pipeline, 
        elastic_params, 
        cv=kfold, 
        scoring='neg_mean_squared_error'
    )
    
    # fitting
    elastic_grid.fit(X_train, y_train)
    best_elastic = elastic_grid.best_estimator_
    y_pred = best_elastic.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    joblib.dump(best_elastic, './best_elastic_model.pkl')
    
    return best_elastic, y_pred, mse, rmse, r2


def estimate_treds(ames):
    
    target = 'SalePrice'
    y = ames[target]
    
    ig_cols = ['PID','SalePrice','GeoRefNo','Prop_Addr','Latitude','Longitude','Date',
               'Risk_RoofMatl','Risk_Exterior1st','Risk_Exterior2nd','Risk_MasVnrType','Risk_WoodDeckSF',
               'BuildingPricePerTotalSF','BuildingValue', 'LandValue', 'TotalSF']
    X = ames.drop(columns=ig_cols)
    
    # 학습/테스트 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    predicted_model = joblib.load('best_elastic_model.pkl')
    
    predicted = predicted_model.predict(X_test)
    
    test_df = pd.DataFrame(
        {'Test_ID': list(y_test.index),
         'Predicted': predicted,
         'Truth': y_test,
         'Resid': predicted - y_test})
    
    _ = plt.figure(figsize=(8, 5))
    _ = plt.scatter(test_df['Truth'], test_df['Predicted'], alpha=0.7, color='royalblue', label='Predicted vs Truth')
    _ = plt.plot([min(test_df['Truth']), max(test_df['Truth'])], [min(test_df['Predicted']), max(test_df['Predicted'])], 'r--', label='Estimated Line')
    _ = plt.xlabel('Truth')
    _ = plt.ylabel('Predicted')
    _ = plt.title('Trends of Residual')
    _ = plt.legend()
    _ = plt.grid(True)
    _ = plt.tight_layout()
    _ = plt.show()
    
def cal_rate_criterion(level):
    if level ==1:
        return 0.95
    elif level ==2:
        return 0.975
    elif level ==3:
        return 1       
    elif level ==4:
        return 1.025
    else:
        return 1.05

def estimate_bill_price(ames, estimate_type, **kargs):
    
    target = 'SalePrice'
    
    ames_raw = pd.read_csv('../data/ames.csv')
    ames_raw_one = ames_raw.head(1) 
    
    ames_raw_one = dataloader.make_risk_point(ames_raw_one)
    ames_raw_one = dataloader.SF_calculator(ames_raw_one)
    
    y = ames_raw_one[target]
    ig_cols = ['PID','SalePrice','GeoRefNo','Prop_Addr','Latitude','Longitude','Risk_RoofMatl','Risk_Exterior1st','Risk_Exterior2nd','Risk_MasVnrType','Risk_WoodDeckSF']
    X = ames_raw_one.drop(columns=ig_cols)
    
    predicted_model = joblib.load('best_elastic_model.pkl')
    predicted_model.get_params
    predicted = predicted_model.predict(X)
    
    
    # if estimate_type == 'single':
    #     ames = dataloader.make_risk_point(ames)
    #     ames = dataloader.SF_calculator(ames)
    predicted_model = joblib.load('best_elastic_model.pkl')
    predicted_model.get_params
    #     y = ames[target]
    #     ig_cols = ['PID','SalePrice','GeoRefNo','Prop_Addr','Latitude','Longitude','Risk_RoofMatl','Risk_Exterior1st','Risk_Exterior2nd','Risk_MasVnrType','Risk_WoodDeckSF']
    #     X = ames.drop(columns=ig_cols)
    #     estimate = predicted_model.predict(X)

    # else:
    # 지도 시각화

    import plotly.express as px


    dataloader = DataLoader()
    ames = dataloader.load_data()

    ratio_additional = 0.003
    overall_cal_rate = ames['Risk_Level'].apply(cal_rate_criterion)
    estimate = (ames['SalePrice'] * ratio_additional) * overall_cal_rate 
    estimate = pd.Series(estimate, name='Estimate')
    ames_estimate = ames.join(estimate)

    cons_cols = ['Latitude', 'Longitude', 'Estimate']
    vis_gis_ames = ames_estimate.loc[:, cons_cols]
    vis_gis_ames = vis_gis_ames.dropna().reset_index(drop=True)


    # Estimate의 최소, 최대값 가져오기
    min_est = vis_gis_ames["Estimate"].min()
    max_est = vis_gis_ames["Estimate"].max()

    # 지도 시각화
    fig = px.scatter_mapbox(
        vis_gis_ames,
        lat="Latitude",
        lon="Longitude",
        color="Estimate",
        color_continuous_scale=[[0, 'rgba(255, 0, 0, 0.2)'], [1, 'rgba(255, 0, 0, 1)']],
        range_color=(min_est, max_est),
        zoom=10,
        height=600
    )

    # 마커 크기 작게 조정
    fig.update_traces(marker=dict(size=5))

    fig.update_layout(
        mapbox_style="open-street-map",
        title="Estimate Map Visualization",
        margin={"r":0,"t":30,"l":0,"b":0}
    )

    fig.show()
        
    return estimate

def insurance_fee(ames):
    
    dataloader = DataLoader()
    ames = dataloader.load_data()
    
    ratio_additional = 0.003
    overall_cal_rate = ames['Risk_Level'].apply(cal_rate_criterion)
    estimate = (ames['SalePrice'] * ratio_additional) * overall_cal_rate 
    estimate = pd.Series(estimate, name='Estimate')
    
    x_1_ticks_range = np.linspace(0.002, 0.004, len(ames))
    
    estimate.sum()
    estimate.sum() /ames['SalePrice'].sum() * 0.8 
    
    mean_estimate = estimate.mean()
    mean_sale_price = (ames['SalePrice'].mean() * 80) / 100
    
    estimate.sum() / ames['SalePrice'].sum()
    estimate.mean() / ames['SalePrice'].mean()

    print("비율(총합 기준):", estimate.sum() / ames['SalePrice'].sum())
    print("비율(평균 기준):", estimate.mean() / ames['SalePrice'].mean())

    fig, ax = plt.subplots(figsize=(8,5))
    _ = ax.plot(x_1_ticks_range, [mean_estimate]*len(x_1_ticks_range), color='red', label='mean estimate')
    _ = ax.plot(x_1_ticks_range, mean_sale_price*x_1_ticks_range, color='blue', label='mean sale price')
    _ = ax.axvline(x=0.00368, color='blue', linestyle='--')
    _ = ax.set_xlabel('Fire Claim Rate')
    _ = ax.set_ylabel('Estimate')
    _ = ax.set_title('Break Even of Estimate by Fire Claim')
    # _ = ax.set_xticks(range(len(mean_sale_price)))
    # _ = ax.set_xticklabels(cnt_RiskLevel.index.astype(str))
    _ = ax.grid(axis='y', linestyle='--', alpha=0.5)
    _ = plt.xticks(rotation=0)
    _ = plt.legend()

    

def main(ames):
    print("===== 화재 발생 시 예상 피해액 모델링 =====\n")
    
    dataloader = DataLoader()
    ames = dataloader.load_data()
    
    # 피해액 책정 모델 구축
    best_model, predictions, mse, rmse, r2  = price_prediction_models(ames)
    best_model.get_params()
    joblib.dump(best_model, 'best_elastic_model.pkl') 
    
    print(f"ELASTIC")
    print(f"RMSE: {rmse:.2f}")
    print(f"R^2: {r2:.4f}")
    
if __name__ == "__main__":

    dataloader = DataLoader()
    dataset = dataloader.load_data()
    
    # risk_columns = [c for c in dataset.columns if c.split('_')[0] == 'Risk']
    # risk_columns
    
    main(dataset)
