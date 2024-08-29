import numpy as np
import pandas as pd
from scipy.stats import shapiro,probplot
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.stattools import adfuller


#函数部分
#读取excel部分列数据
#     参数:
#     excel_name(str): excel表的名称，最好选用excel表
#     sheet_name(str): 选择sheet的名称
#     use_data(str): 选择列的范围
#     skip_num(num): 列跳过的数字
#     index(str): 索引依据（x轴）
#     content(str):索引内容（y轴）
#     返回:
#     返回值(pandas.core.series.Series): 返回一个pandas的series对象
#     示例:
#     ts=read_excel_data('08.26.xlsx','运营日报-SEA苹果端','A,F',1,'Unnamed: 0','ARPU')
def read_excel_data(excel_name,sheet_name,use_data,skip_num,index,content):
    ts = pd.read_excel(excel_name, sheet_name=sheet_name, usecols=use_data, skiprows=skip_num, na_filter=True)
    ts[index] = pd.to_datetime(ts[index])
    ts.set_index(index, inplace=True)
    arpu_series = ts[content]
    return arpu_series
#将数据分为训练集和测试集
#     参数:
#     data(pandas.core.series.Series): 要分割对的数据集
#     num(num): 选择sheet的名称
#     返回:
#     [train,test](arr): 返回一个数组，包含train和test
#     示例:
#     ARPU_data=data_partiton(ts,0.8)
def data_partiton(data,num):
    train_size=int(len(data)*num)
    train,test=data[:train_size], data[train_size:]
    return [train,test]
# 数据预处理：删除数据缺失值
#     参数:
#     data(pandas.core.series.Series): 要处理的数据集
#     返回:
#     cleaned_data(pandas.core.series.Series): 返回一个处理过的数据集
#     示例:
#     clearData=data_clear(ARPU_data[0])
def data_clear(data):
    cleaned_data = data.dropna()
    return cleaned_data
# 数据预处理：差分数据
#     参数:
#     data(pandas.core.series.Series): 要处理的数据集
#     返回:
#     diff_data(pandas.core.series.Series): 返回一个处理过的数据集
#     示例:
#     diffData=data_diff(clearData)
def data_diff(data):
    diff_data = data.diff()
    return diff_data
# 数据预处理：对数据进行 ADF检验观测平稳
#     参数:
#     data(pandas.core.series.Series): 要处理的数据集
#     返回:
#     adf_result(arr):返回一个包含检测结果的数组
#     示例:
#     adf_test(clearData)
def adf_test(data):
    adf_result = adfuller(data_clear(data), autolag='AIC')
    print('ADF Statistic:', adf_result[0])
    print('p-value:', adf_result[1])
    print('Critical Values:', adf_result[4])
    return adf_result
# 参数选择：通过最小aic来选择范围内最适合的p、d、q参数
#     参数:
#     p_num(num):用于p参数的最大循环数
#     data(pandas.core.series.Series): 要处理的数据集
#     返回:
#     null
#     示例:
#     loop_select_param(3,3,3,data)
def loop_select_param(p_num,d_num,q_num,data):
    best_aic = float('inf')
    best_order = None
    for p in range(p_num):
        for d in range(d_num):
            for q in range(q_num):
                try:
                    model = ARIMA(data, order=(p, d, q))
                    model_fit = model.fit()
                    aic = model_fit.aic
                    if aic < best_aic:
                        best_aic = aic
                        best_order = (p, d, q)
                except:
                    continue
    print('最佳 ARIMA 模型参数:', best_order)
# 参数选择：构建ARIMA模型并拟合
#     参数:
#     p_num(num):用于p参数的最大循环数
#     data(pandas.core.series.Series): 要处理的数据集
#     返回:
#     model_fit(pandas.core.series.Series): 被处理的数据集
#     示例:
#     ARIMA_fit(data)
def ARIMA_fit(p_num,d_num,q_num,data):
    model=ARIMA(data, order=(p_num,d_num,q_num))  #设置筛选范围内上面的最佳参数
    model_fit=model.fit()
    return model_fit
# 修正ARIMA模型数据
#     参数:
#     fitData(pandas.core.series.Series): 要处理的数据集
#     partDataArr(arr):分割的数据集数组
#     返回:
#     forecast_series(pandas.core.series.Series): 被处理的数据集
#     示例:
#     forecast_data_process(fitData,ARPU_data)
def forecast_data_process(fitData,partDataArr):
    forecast = fitData.forecast(steps=len(partDataArr[1]))
    forecast_index=pd.date_range(start=partDataArr[0].index[-1]+pd.DateOffset(days=1), periods=len(partDataArr[1]), freq='D')
    forecast_series = pd.Series(forecast, index=forecast_index)
    return forecast_series
#Ljung_Box检测
#     参数:
#     residuals(pandas.core.series.Series): 误差
#     lag_num(num): lag数量
#     返回:
#     null
#     示例:
#     check_Ljung_Box(residuals,12)
def check_Ljung_Box(residuals,lag_num):
    lb_test = sm.stats.diagnostic.acorr_ljungbox(residuals, lags=[lag_num], return_df=True)
    print("Ljung-Box Test:")
    print(lb_test)
#shapiro检测
#     参数:
#     residuals(pandas.core.series.Series): 误差
#     返回:
#     null
#     示例:
#     check_shapiro(residuals)
def check_shapiro(residuals):
    stat, p_value = shapiro(residuals)
    print(f'\nShapiro-Wilk Test: Statistics={stat}, p-value={p_value}')
# shapiro检测
#     参数:
#     dataArr(arr): 数据数组
#     返回:
#     null
#     示例:
#     cal_err_index([ARPU_data[1],fitData.forecast(steps=len(ARPU_data[1]))])
def cal_err_index(dataArr):
    mse = mean_squared_error(dataArr[0], dataArr[1])
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(dataArr[0], dataArr[1])
    print(f'MSE: {mse}')
    print(f'RMSE: {rmse}')
    print(f'MAE: {mae}')
#将数据绘制对应位置的图
#     参数:
#     dataArr(arr): 数据数组
#     lag_num(num): lag数量
#     返回:
#     null
#     示例:
#     plot_ready([clearData,forecastSeries],12)
def plot_ready(dataArr,lag_num):
    plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(2, 2)
    ax1=plt.subplot(gs[0, :2])
    ax1.plot(dataArr[0], label='origin data', color='blue')
    ax1.plot(dataArr[1], label='forecast data', color='red')
    ax1.set_title('forecast-ARPU')
    ax1.set_xlabel('time')
    ax1.set_ylabel('ARPU')
    # 样本数据够可以考虑pacf acf
    # ax2=plt.subplot(gs[1, 0])
    # plot_acf(dataArr[0], lags=lag_num, ax=ax2)
    # ax2.set_title('ACF img')
    # ax2=plt.subplot(gs[1, 0])
    # plot_pacf(dataArr[0], lags=lag_num, ax=ax2)
    # ax2.set_title('PACF img')
    if len(dataArr)>2:
        ax4=plt.subplot(gs[1,:2])
        probplot(dataArr[2], dist="norm", plot=plt)
        ax4.set_title('Q-Q Plot of Residuals')
    plt.tight_layout()
    plt.show()

#调用部分
ts=read_excel_data('8.25.xlsx','运营日报','A,F',1,'Unnamed: 0','ARPU')
ARPU_data=data_partiton(ts,0.8)
clearData=data_clear(ARPU_data[0])
diffData=data_diff(clearData)
adfResult=adf_test(clearData)
# loop_select_param(3,3,3,clearData)   #性能消耗较大
fitData=ARIMA_fit(2,1,1,clearData)
forecastSeries=forecast_data_process(fitData,ARPU_data)
residuals = fitData.resid
check_Ljung_Box(residuals,12)
check_shapiro(residuals)
cal_err_index([ARPU_data[1],fitData.forecast(steps=len(ARPU_data[1]))])
plot_ready([clearData,forecastSeries,residuals,ts],12)



