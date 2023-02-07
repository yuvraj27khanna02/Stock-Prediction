# Stock-Prediction
Predict future value of TATACOMM (Tata Communications Limited) stock based on previous data

## Libraries
pandas, numpy, matplotlib, sklearn

## Description
TATACOMM (Tata Communications Limited) "Close" value predicted using Random Forest Regression Model.  The model is trained on TATACOMM stock prices from 1996 to 2021.  Train:Test data split is 8:2.  To train the model the n_estimators is set to 1000, min sample split is set to 2, min sample lead is set to 1, max depth is set to 13, and bootstrap is set to True.  You can run the program on your own and get predictions in a csv format.

## How to run program
Run StockModel.py to run the program.  The verbose of training the model can be turned off by setting VERBOSE_VALUE = 0 on line 9.  input "y" to get predictions for one year, "m" to get predictions for one month, or "d" to get predictions for 5 days when prompted.  the chosen predicted data will be saved in a csv format.

### NOTICE
This program is a Work In Progress and all feedback on it is appreciated.

## If you cannot run this program on your device
My notebook on research collab is https://colab.research.google.com/drive/1ZW0_uXjIYhQvHmQG8jXKTTa7QRFQWMe2?usp=sharing .  Run the first code cell to run the program.  Feel free to ;eave a comment.