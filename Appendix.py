import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

data=pd.read_csv('data.csv')

features = ['p1_momentum', 'p2_momentum', 'score_gap']
target = 'p1_momentum' 

df_lstm = data[features + [target]]

# Normalize
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_lstm)

# Split data into input (x) and target (y)
X = scaled_data[:, :-1]
y = scaled_data[:, -1]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape input features for LSTM model
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Build LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model on the test set
loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")

# Make predictions
predictions = model.predict(X_test)

# Inverse transform the predictions and y_test to get them in the original scale
predictions = scaler.inverse_transform(np.hstack((X_test.reshape(X_test.shape[0], X_test.shape[2]), predictions)))
y_test = scaler.inverse_transform(np.hstack((X_test.reshape(X_test.shape[0], X_test.shape[2]), y_test.reshape(-1, 1))))

# Define a threshold
threshold = 20

# Calculate the difference between consecutive momentum values in y_test
momentum_changes = np.abs(np.diff(y_test[:, -1]))

# Identify the indices where the momentum change exceeds the threshold
change_indices = np.where(momentum_changes > threshold)[0]

# Calculate R-squared
r2 = r2_score(y_test[:, -1], predictions[:, -1])

# Find the index of the maximum momentum change
max_change_index = change_indices[np.argmax(momentum_changes[change_indices])]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(predictions[:, -1], label='Predicted Momentum', color='blue')
plt.plot(y_test[:, -1], label='Actual Momentum', color='green')

# Mark the points with momentum changes in orange
plt.scatter(change_indices, predictions[change_indices, -1], color='orange', label='Swings')

# Mark the point with maximum momentum change in red
plt.scatter(max_change_index, predictions[max_change_index, -1], color='red', label='Turning Point')

# Add title and labels
plt.title('Actual vs Predicted Momentum\n(Red dots represent Swings)\nR-squared: {:.2f}'.format(r2))
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()
