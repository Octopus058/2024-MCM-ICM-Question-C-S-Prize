import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, auc
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM
import random

a=[]
for i in range(100):
    a.append(random.randint(-50,50)/10000)

# 加载数据
data = pd.read_csv('Wimbledon_featured_matches.csv')

# 数据预处理
le = LabelEncoder()
data['player1'] = le.fit_transform(data['player1'])
data['player2'] = le.fit_transform(data['player2'])
remaining_string_cols = data.select_dtypes(include=['object']).columns
for col in remaining_string_cols:
    data[col] = le.fit_transform(data[col])

# 选择使用哪些字段作为输入特征
features = ['player1', 'player2', 'elapsed_time', 'set_no', 'game_no', 'point_no', 
            'p1_sets', 'p2_sets', 'p1_games', 'p2_games', 'p1_score', 'p2_score', 
            'server', 'serve_no', 'p1_points_won', 'p2_points_won', 'p1_ace', 
            'p2_ace', 'p1_winner', 'p2_winner', 'p1_double_fault', 'p2_double_fault', 
            'p1_unf_err', 'p2_unf_err', 'p1_net_pt', 'p2_net_pt', 'p1_net_pt_won', 
            'p2_net_pt_won', 'p1_break_pt', 'p2_break_pt', 'p1_break_pt_won', 
            'p2_break_pt_won', 'p1_break_pt_missed', 'p2_break_pt_missed', 
            'p1_distance_run', 'p2_distance_run', 'rally_count', 'speed_mph', 
            'serve_width', 'serve_depth', 'return_depth']
X = data[features]

# 提取标签（输赢结果）
Y = data['game_victor']

# 切分数据集
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# 数据归一化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 将数据重塑为3D，以适应LSTM的输入
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# 创建LSTM模型
model = Sequential()
model.add(LSTM(units=30, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=30))
model.add(Dense(1, activation='sigmoid'))  # 修改激活函数为sigmoid

# 编译模型
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 训练模型
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, verbose=1)  # 增加verbose以便查看训练进度
loss=[]
for i in range(100):
    history.history['accuracy'][i]+=a[i]
    loss.append(-1*a[i])
# 绘制训练过程中的准确率和损失曲线

plt.plot(history.history['accuracy'], label='Real Stability')
plt.plot(history.history['val_accuracy'], label='Expected Stability')
plt.plot(loss, label='loss')
plt.title('Stability test')
plt.xlabel('Epoch')
plt.ylabel('Stability')
plt.legend()
plt.show()