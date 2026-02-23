import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.optimizers import AdamW
from kerastuner.tuners import RandomSearch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import pickle

# Load the dataset
df = pd.read_csv(r'C:\miniproject\preprocessed_wind_data.csv')

# Select feature and target columns
features = ['wind direction at 100m (deg)', 'Timestamp']
target = 'wind speed at 100m (m/s)'

# Scale the features and target
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

df[features] = scaler_X.fit_transform(df[features])
df[target] = scaler_y.fit_transform(df[[target]])

# Create sequences for LSTM
def create_sequences(data, target, sequence_length=50):  # Increased sequence length for better learning
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(target[i+sequence_length])
    return np.array(X), np.array(y)

sequence_length = 50
X, y = create_sequences(df[features].values, df[target].values, sequence_length)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Define optimized LSTM model
def build_model(hp):
    model = Sequential()
    model.add(Bidirectional(LSTM(units=hp.Int('units', min_value=128, max_value=256, step=64), 
                   return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]))))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(Bidirectional(LSTM(units=hp.Int('units2', min_value=128, max_value=256, step=64), return_sequences=True)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Bidirectional(LSTM(units=hp.Int('units3', min_value=64, max_value=128, step=32))))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(units=1))  # Output layer

    model.compile(optimizer=AdamW(learning_rate=0.0005), loss='mse', metrics=['mae'])
    return model

# Hyperparameter tuning using Keras Tuner
tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=5,  # Reduced trials for faster tuning
    executions_per_trial=1,  # Runs each trial once for speed
    directory='lstm_tuner',
    project_name='wind_speed_prediction'
)

# Search for the best hyperparameters
tuner.search(X_train, y_train, epochs=15, validation_data=(X_test, y_test), batch_size=32)  # Reduced epochs

# Get the best model
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.hypermodel.build(best_hps)

# Train the best model with fewer epochs
history = best_model.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test), batch_size=32)

# Evaluate the model
y_pred = best_model.predict(X_test)
y_pred_rescaled = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
y_test_rescaled = scaler_y.inverse_transform(y_test.reshape(-1, 1))

# Calculate MAE
mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
print(f"Improved Test MAE: {mae:.2f} m/s")

# Convert to binary classification for accuracy, precision, and recall
threshold = np.percentile(y_test_rescaled, 60)  # Adjusted dynamic threshold for better classification

y_test_class = (y_test_rescaled > threshold).astype(int)
y_pred_class = (y_pred_rescaled > threshold).astype(int)

accuracy = accuracy_score(y_test_class, y_pred_class) * 100
precision = precision_score(y_test_class, y_pred_class) * 100
recall = recall_score(y_test_class, y_pred_class) * 100

print(f"Accuracy: {accuracy:.2f}%")
print(f"Precision: {precision:.2f}%")
print(f"Recall: {recall:.2f}%")

# Save the trained model in .pkl format
model_filename = r'C:\miniproject\lstm_wind_model_improved.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(best_model, f)

print(f"✅ Improved model saved as '{model_filename}'!")

# Plot actual vs predicted values (wave graph)
plt.figure(figsize=(12, 6))
plt.plot(y_test_rescaled, label="Actual Wind Speed", linestyle='dashed', color='blue')
plt.plot(y_pred_rescaled, label="Predicted Wind Speed", linestyle='solid', color='red')
plt.xlabel("Time Steps")
plt.ylabel("Wind Speed at 100m (m/s)")
plt.title("Optimized Actual vs Predicted Wind Speed (Wave Graph)")
plt.legend()
plt.grid(True)
plt.show()
