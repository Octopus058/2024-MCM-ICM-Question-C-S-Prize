import pandas as pd
from sklearn.preprocessing import LabelEncoder
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split

#读入数据
df = pd.read_csv('0.csv')


#数据清洗
def dataProcess(data):
    columns_to_encode = data.columns[-3:]
    label_encoder = LabelEncoder()
    for column in columns_to_encode:
        data[column] = label_encoder.fit_transform(data[column])
    replace_dict = {'AD':pd.NA, 'F': 1, 'B' : 2}
    data['p1_score'] = pd.to_numeric(data['p1_score'], errors = 'coerce')
    data['p2_score'] = pd.to_numeric(data['p2_score'], errors = 'coerce')
    data.replace({'winner_shot_type':replace_dict}, inplace = True)
    data.drop_duplicates(inplace = True)
    data = data_clean(data)
    data['score_gap'] = data['p1_score'] - data['p2_score']
    return data
#填补空白和异常值
def data_clean(data):
    data = data.drop_duplicates()
    data = data.fillna(method = 'ffill')
    data = data.fillna(method = 'bfill')
    numeric_columns = data.select_dtypes(include = ['number']).columns
    for col in numeric_columns:
        col_mean = data[col].mean()
        col_std = data[col].std()
        lower_bound = col_mean - 3 * col_std
        upper_bound = col_mean + 3 * col_std
        data.loc[(data[col] < lower_bound)| (data[col] > upper_bound),col] = None
    data = data.fillna(method = 'ffill')
    data = data.fillna(method = 'bfill')
    return data

df = dataProcess(df)

def calculate_momentum(data):
    data['server'] = data['server'] if data['server'] else 70
    data['server'] = 100 if data['server'] else 70
    # 基础动量计算
    base_momentum1 = ((data['p1_score']/data['game_no'] * 5 + data['p1_games'] * 100 + data['server'] + data['score_gap'] * 5) * 0.25) * 0.45
    base_momentum2 = ((data['speed_mph'] + data['p1_score'] + data['p1_winner']) * 0.333) * 0.55
    # 创造动量计算
    change_momentum = (data['p1_ace'] - data['p1_double_fault'] - data['p1_unf_err']) * 2 * 0.333
    # 总动量是上述三部分之和
    data['p1_momentum'] = base_momentum1 + change_momentum + base_momentum2
    data['server'] = data['server'] if data['server'] else 50
    data['server'] = 100 if data['server'] else 50
    # 基础动量计算
    base_momentum1 = ((data['p2_score']/data['game_no'] * 5 + data['p2_games'] * 100 + data['server'] - data['score_gap'] * 5) * 0.25) * 0.45
    base_momentum2 = ((data['speed_mph'] + data['p2_score'] + data['p2_winner']) * 0.333) * 0.55
    # 创造动量计算
    change_momentum = (data['p2_ace'] - data['p2_double_fault'] - data['p2_unf_err']) * 2 * 0.333
    data['p2_momentum'] = base_momentum1 + change_momentum + base_momentum2
    return data
      
data = df.apply(calculate_momentum, axis=1)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#随机森林

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 选择特征和目标变量
features = data[['score_gap', 'p1_momentum']]
target = data['p1_momentum']

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# 初始化随机森林模型
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# 训练模型
rf_model.fit(X_train, y_train)

# 模型评估
y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# 计算并打印R^2分数
r2 = r2_score(y_test, y_pred)
print("R^2 Score:", r2)

# 预测新数据
new_data = data[['score_gap', 'p1_momentum']].tail(1)  # 假设新数据为最后一行
predicted_value = rf_model.predict(new_data)
print("Predicted Value:", predicted_value)


# 决策树
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 选择特征和目标变量
features = data[['score_gap', 'p1_momentum']]
target = data['p1_momentum']

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# 初始化决策树回归模型
dt_model = DecisionTreeRegressor(random_state=42)

# 训练模型
dt_model.fit(X_train, y_train)

# 模型评估
y_pred = dt_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# 计算并打印R^2分数
r2 = r2_score(y_test, y_pred)
print("R^2 Score:", r2)

# 预测新数据
new_data = data[['score_gap', 'p1_momentum']].tail(1)  # 假设新数据为最后一行
predicted_value = dt_model.predict(new_data)
print("Predicted Value:", predicted_value)


# 导入必要的库
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd

# 线性回归

# 选择特征和目标变量
features = data[['score_gap', 'p1_momentum']]
target = data['p1_momentum']

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# 初始化线性回归模型
linear_model = LinearRegression()

# 训练模型
linear_model.fit(X_train, y_train)

# 计算并打印R^2分数
r2 = r2_score(y_test, y_pred)
print("R^2 Score:", r2)

# 模型评估
y_pred = linear_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# 预测新数据
new_data = data[['score_gap', 'p1_momentum']].tail(1)  # 替换new_score_gap_value为实际值
predicted_value = linear_model.predict(new_data)
print("Predicted Value:", predicted_value)

data1 = data['score_gap']
data2 = data['p1_momentum']
data3 = data['p2_momentum']
corr1, p_value1 = stats.pearsonr(data1,data2)
corr2, p_value2 = stats.pearsonr(data1,data3)

plt.figure(figsize = (10,6))
plt.title('Momentum & Score Gap Over Time')
plt.xlabel('Time')
plt.ylabel('Value')
plt.plot(data['score_gap'][:300], label = 'score gap')
plt.plot(data['p1_momentum'][:300], label = 'A')
plt.plot(data['p2_momentum'][:300], label = 'B')
plt.legend()
plt.grid(True)
plt.show()

# Assuming 'data' DataFrame already has 'p1_momentum', 'p2_momentum', and 'score_gap' columns
features = ['p1_momentum', 'p2_momentum', 'score_gap']
target = 'p1_momentum'  # Replace 'your_target_column' with the actual column you want to predict

# Select relevant features and target
df_lstm = data[features + [target]]

# Normalize the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_lstm)

# Split data into input (X) and target (y)
X = scaled_data[:, :-1]  # Input features (p1_momentum, p2_momentum, score_gap)
y = scaled_data[:, -1]   # Target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape input features for LSTM model
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Build LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))  # Output layer with 1 neuron for regression
model.compile(optimizer='adam', loss='mse')  # Mean Squared Error (MSE) loss for regression problems

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model on the test set
loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")

# Make predictions
predictions = model.predict(X_test)

# Inverse transform the predictions and y_test to get them in the original scale
predictions = scaler.inverse_transform(np.hstack((X_test.reshape(X_test.shape[0], X_test.shape[2]), predictions)))
y_test = scaler.inverse_transform(np.hstack((X_test.reshape(X_test.shape[0], X_test.shape[2]), y_test.reshape(-1, 1))))

# Define a threshold for momentum changes
threshold = 20

# Calculate the difference between consecutive momentum values in y_test
momentum_changes = np.abs(np.diff(y_test[:, -1]))

# Identify the indices where the momentum change exceeds the threshold
change_indices = np.where(momentum_changes > threshold)[0]

from sklearn.metrics import r2_score

# Calculate R-squared
r2 = r2_score(y_test[:, -1], predictions[:, -1])

# Find the index of the maximum momentum change
max_change_index = change_indices[np.argmax(momentum_changes[change_indices])]

# Plot the actual vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(predictions[:, -1], label='Predicted Momentum', color='blue')
plt.plot(y_test[:, -1], label='Actual Momentum', color='green')

# Mark the points with momentum changes in yellow
plt.scatter(change_indices, predictions[change_indices, -1], color='orange', label='Swings')

# Mark the point with maximum momentum change in red
plt.scatter(max_change_index, predictions[max_change_index, -1], color='red', label='Turning Point')

plt.title('Actual vs Predicted Momentum\n(Red dots represent Swings)\nR-squared: {:.2f}'.format(r2))
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()
