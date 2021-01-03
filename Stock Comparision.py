import pandas as pd
import matplotlib
from pylab import rcParams
from datetime import date

# Load Data
df = pd.read_csv("NIFTY 50.csv")

# Convert Date column to datetime
df.loc[:, 'Date'] = pd.to_datetime(df['Date'],format='%Y-%m-%d')

# Change all column headings to be lower case, and remove spacing
df.columns = [str(x).lower().replace(' ', '_') for x in df.columns]

# Sort by datetime
df.sort_values(by='date', inplace=True, ascending=True)

# Plot adjusted close over time
rcParams['figure.figsize'] = 8, 8 # width 10, height 8

ax = df.plot(x='date', y='close', style='b-', grid=True)
ax.set_xlabel("Year")
ax.set_ylabel("Nifty Index")

# Get sizes of each of the datasets
num_cv = int(0.2*len(df))
num_test = int(0.2*len(df))
num_train = len(df) - num_cv - num_test
print("num_train = " + str(num_train))
print("num_cv = " + str(num_cv))
print("num_test = " + str(num_test))

# Split into train, cv, and test
train = df[:num_train].copy()
cv = df[num_train:num_train+num_cv].copy()
train_cv = df[:num_train+num_cv].copy()
test = df[num_train+num_cv:].copy()
print("train.shape = " + str(train.shape))
print("cv.shape = " + str(cv.shape))
print("train_cv.shape = " + str(train_cv.shape))
print("test.shape = " + str(test.shape))

# Plot adjusted close over time
rcParams['figure.figsize'] = 8, 8 # width 10, height 8
matplotlib.rcParams.update({'font.size': 14})

ax = train.plot(x='date', y='close', style='b-', grid=True)
ax = cv.plot(x='date', y='close', style='y-', grid=True, ax=ax)
ax = test.plot(x='date', y='close', style='g-', grid=True, ax=ax)
ax.legend(['train', 'validation', 'test'])
ax.set_xlabel("Year")
ax.set_ylabel("Nifty Index")

# Compare various methods
results_dict = {'Method': ['Last Value', 'Moving Average', 'Linear Regression', 'XGBoost', 'LSTM'],
                'RMSE': [1.108, 1.211, 1.324, 1.109, 1.241],
                'MAPE(%)': [0.697, 0.803, 0.877, 0.702, 0.845]}
results = pd.DataFrame(results_dict)
print(results)

# Read all dataframes for the different methods
test_last_value = pd.read_csv("test_last_value.csv", index_col=0)
test_last_value.loc[:, 'date'] = pd.to_datetime(test_last_value['date'],format='%Y-%m-%d')

test_mov_avg = pd.read_csv("test_mov_avg.csv", index_col=0)
test_mov_avg.loc[:, 'date'] = pd.to_datetime(test_mov_avg['date'],format='%Y-%m-%d')

test_lin_reg = pd.read_csv("test_lin_reg.csv", index_col=0)
test_lin_reg.loc[:, 'date'] = pd.to_datetime(test_lin_reg['date'],format='%Y-%m-%d')

test_xgboost = pd.read_csv("test_xgboost.csv", index_col=0)
test_xgboost.loc[:, 'date'] = pd.to_datetime(test_xgboost['date'],format='%Y-%m-%d')

test_lstm = pd.read_csv("test_lstm.csv", index_col=0)
test_lstm.loc[:, 'date'] = pd.to_datetime(test_lstm['date'],format='%Y-%m-%d')

# Plot all methods together to compare
rcParams['figure.figsize'] = 8, 8 # width 10, height 8
matplotlib.rcParams.update({'font.size': 14})

ax = df.plot(x='date', y='close', style='g-', grid=True)
ax = test_last_value.plot(x='date', y='est_N1', style='r-', grid=True, ax=ax)
ax = test_mov_avg.plot(x='date', y='est_N2', style='b-', grid=True, ax=ax)
ax = test_lin_reg.plot(x='date', y='est_N5', style='m-', grid=True, ax=ax)
ax = test_xgboost.plot(x='date', y='est', style='y-', grid=True, ax=ax)
ax = test_lstm.plot(x='date', y='est', style='c-', grid=True, ax=ax)
ax.legend(['Test', 
           'Predictions using Last Value', 
           'Predictions using Moving Average', 
           'Predictions using Linear Regression',
           'Predictions using XGBoost',
           'Predictions using LSTM'], loc='lower left')
ax.set_xlabel("Date")
ax.set_ylabel("Nifty Index")

rcParams['figure.figsize'] = 8, 8 # width 10, height 8
matplotlib.rcParams.update({'font.size': 14})

ax = test.plot(x='date', y='close', style='g-', grid=True)
ax = test_last_value.plot(x='date', y='est_N1', style='r-', grid=True, ax=ax)
ax = test_mov_avg.plot(x='date', y='est_N2', style='b-', grid=True, ax=ax)
ax = test_lin_reg.plot(x='date', y='est_N5', style='m-', grid=True, ax=ax)
ax = test_xgboost.plot(x='date', y='est', style='y-', grid=True, ax=ax)
ax = test_lstm.plot(x='date', y='est', style='c-', grid=True, ax=ax)
ax.legend(['Test', 
           'Predictions using Last Value', 
           'Predictions using Moving Average', 
           'Predictions using Linear Regression',
           'Predictions using XGBoost',
           'Predictions using LSTM'], loc='lower left')
ax.set_xlabel("Date")
ax.set_ylabel("Nifty Index")
ax.set_title("Test")

rcParams['figure.figsize'] = 8, 8 # width 10, height 8
matplotlib.rcParams.update({'font.size': 14})

ax = test.plot(x='date', y='close', style='g-', grid=True)
ax = test_last_value.plot(x='date', y='est_N1', style='r-', grid=True, ax=ax)
ax = test_mov_avg.plot(x='date', y='est_N2', style='b-', grid=True, ax=ax)
ax = test_lin_reg.plot(x='date', y='est_N5', style='m-', grid=True, ax=ax)
ax = test_xgboost.plot(x='date', y='est', style='y-', grid=True, ax=ax)
ax = test_lstm.plot(x='date', y='est', style='c-', grid=True, ax=ax)
ax.legend(['Test', 
           'Predictions using Last Value', 
           'Predictions using Moving Average', 
           'Predictions using Linear Regression',
           'Predictions using XGBoost',
           'Predictions using LSTM'], loc='lower left')
ax.set_xlabel("Date")
ax.set_ylabel("Nifty Index")
ax.set_title("Zoom in to Test")
ax.set_xlim([date(2020, 1, 1), date(2020, 5, 8)])
ax.set_ylim([7000, 13000])