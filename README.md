# Developed By : Pravin Raj A
# Register No. : 212222240079

# Ex.No: 07                                       AUTO REGRESSIVE MODEL
### Date: 



### AIM:
To Implementat an Auto Regressive Model using Python
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error

#### Read the CSV file into a DataFrame
data = pd.read_csv("/content/BTC-USD(1).csv")  

#### Parse the date and set it as index
# Change: Assume the column is named 'Date' instead of 'Order Date'
data['Date'] = pd.to_datetime(data['Date'])  
data.set_index('Date', inplace=True)

#### Filter for 'Furniture' sales
# Change: Since this is Bitcoin Data, removing sales category
# furniture_sales = data[data['Category'] == 'Furniture']
# Change: Use the 'Close' column to get bitcoin values
bitcoin_values = data['Close']

#### Aggregate monthly sales
# Change: Calculate monthly values instead of sales
monthly_values = bitcoin_values.resample('MS').sum() 

#### Perform Augmented Dickey-Fuller test to check stationarity
result = adfuller(monthly_values) 
print('ADF Statistic:', result[0])
print('p-value:', result[1])

#### Split the data into training and testing sets
train_data = monthly_values.iloc[:int(0.8*len(monthly_values))]
test_data = monthly_values.iloc[int(0.8*len(monthly_values)):]

#### Fit an AutoRegressive (AR) model with 13 lags
lag_order = 13
model = AutoReg(train_data, lags=lag_order)
model_fit = model.fit()

#### Plot Autocorrelation Function (ACF)
plot_acf(monthly_values)
plt.title('Autocorrelation Function (ACF) - Bitcoin Values') #Change: title
plt.show()

#### Plot Partial Autocorrelation Function (PACF)
plot_pacf(monthly_values)
plt.title('Partial Autocorrelation Function (PACF) - Bitcoin Values') #Change: title
plt.show()

#### Make predictions using the AR model
predictions = model_fit.predict(start=len(train_data), end=len(train_data)+len(test_data)-1)

#### Compare the predictions with the test data
mse = mean_squared_error(test_data, predictions)
print('Mean Squared Error (MSE):', mse)

#### Plot the test data and predictions
plt.plot(test_data.index, test_data, label='Test Data', color='blue')
plt.plot(test_data.index, predictions, label='Predictions', color='red')
plt.xlabel('Date')
plt.ylabel('Values') #Change: y-axis label
plt.title('AR Model Predictions vs Test Data - Bitcoin Values') #Change: title
plt.legend()
plt.show()

```

### OUTPUT:

#### Augmented Dickey-Fuller test

![image](https://github.com/user-attachments/assets/7f37a79e-303f-4533-9c77-b86caeb4d559)



#### PACF - ACF

![download](https://github.com/user-attachments/assets/1b736281-b31d-4f41-b4dc-c444e0c7663a)

![download](https://github.com/user-attachments/assets/9a1ff647-6a99-4ca1-a6ec-bbd3e33cf470)

#### Mean Squared Error

![image](https://github.com/user-attachments/assets/c4e3ce35-75d1-49ad-b164-2d0a7726cfad)


#### PREDICTION

![download](https://github.com/user-attachments/assets/cf7edddd-1667-40d8-91e1-595811c87aa0)


### RESULT:
Thus we have successfully implemented the auto regression function using python.
