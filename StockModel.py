import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

VERBOSE_VALUE = 4

# setup dataset
df = pd.read_csv("TATACOMM.csv")
TATA_df = df[["Date", "Close", "High", "Low", "Volume"]]
TATA_df.set_index("Date", inplace=True)
TATA_df.dropna(inplace=True)

x_data = TATA_df[["Close", "High", "Low", "Volume"]].values
y_data = TATA_df["Close"].values

# pre processing data
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=2705)

scale = StandardScaler()
x_train = scale.fit_transform(x_train)

# modeling data
TATA_model = RandomForestRegressor(n_estimators=1000,
                                   random_state=30,
                                   min_samples_split=2,
                                   min_samples_leaf=1,
                                   max_depth=13,
                                   bootstrap=True,
                                   verbose=VERBOSE_VALUE)

TATA_model.fit(x_train, y_train)

# predicting data
y_predict = TATA_model.predict(x_test)

# model predicted data MAE, MSE, RMSE, R2, TestScore, Accuracy
TATA_model_MeanAbsoluteError = round(metrics.mean_absolute_error(y_data, y_predict), 4)
TATA_model_MeanSquaredError = round(metrics.mean_squared_error(y_data, y_predict), 4)
TATA_model_RootMeanSquaredError = round(np.sqrt(metrics.mean_squared_error(y_data, y_predict)), 4)
TATA_model_R2 = round(metrics.r2_score(y_data, y_predict), 4)
TATA_model_TestScore = round(TATA_model.score(x_data, y_data) * 100, 2)
TATA_model_Accuracy = 100 - np.mean(100*((abs(y_predict-y_data)/y_data)))

# storing prediction data
TATA_predictions = pd.DataFrame({"Predictions": y_predict},
                                index=pd.date_range(start=df.index[-1], periods=len(y_predict),
                                                    freq="D"))

TATA_one_year_prediction = pd.DataFrame(TATA_predictions[:252])
TATA_one_month_prediction = pd.DataFrame(TATA_predictions[:21])
TATA_five_days_prediction = pd.DataFrame(TATA_predictions[:5])

user_input = input(str("What data prediction do u want to see?"))
if user_input.lower() == "y":
    TATA_one_year_prediction.to_csv("one_year_prediction.csv")
if user_input.lower() == "m":
    TATA_one_month_prediction.to_csv("one_month_prediction.csv")
if user_input.low() == "d":
    TATA_five_days_prediction.to_csv("five_days_prediction.csv")