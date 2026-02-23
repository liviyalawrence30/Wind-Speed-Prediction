import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from kerastuner.tuners import RandomSearch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, precision_score, accuracy_score
import matplotlib.pyplot as plt
import joblib
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
def create_sequences(data, target, sequence_length=10):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(target[i+sequence_length])
    return np.array(X), np.array(y)
sequence_length = 10  # Increased sequence length
X, y = create_sequences(df[features].values, df[target].values, sequence_length)
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
# Define LSTM model builder for hyperparameter tuning
def build_model(hp):
    model = Sequential()
    model.add(LSTM(units=hp.Int('units', min_value=64, max_value=256, step=64), 
                   return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float('dropout', 0.1, 0.5, step=0.1)))
    model.add(LSTM(units=hp.Int('units2', min_value=64, max_value=256, step=64), return_sequences=False))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float('dropout2', 0.1, 0.5, step=0.1)))
    model.add(Dense(units=1))
    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', [0.001, 0.0005, 0.0001])),
                  loss='mse', metrics=['mae'])
    return model
# Hyperparameter tuning using Keras Tuner
tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=5,  # Increased trials for better tuning
    executions_per_trial=1,
    directory='lstm_tuner',
    project_name='wind_speed_prediction'
)
# Search for the best hyperparameters
tuner.search(X_train, y_train, epochs=30, validation_data=(X_test, y_test), batch_size=32)
# Get the best model
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.hypermodel.build(best_hps)
# Callbacks for better training
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# Train the best model
history = best_model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), batch_size=32, 
                         callbacks=[reduce_lr, early_stopping])
# Evaluate the model
test_loss, test_mae = best_model.evaluate(X_test, y_test)
print(f"Test MAE: {test_mae}")
# Predictions
y_pred = best_model.predict(X_test)
y_pred_rescaled = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
y_test_rescaled = scaler_y.inverse_transform(y_test.reshape(-1, 1))
# Calculate MSE
mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
print(f"Mean Squared Error (MSE): {mse}")
# Define a threshold-based accuracy metric
threshold = 0.1  # 10% error tolerance
accuracy = np.mean(np.abs(y_pred_rescaled - y_test_rescaled) / y_test_rescaled < threshold)
print(f"Accuracy (within 10% error margin): {accuracy * 100:.2f}%")
# Save the trained model in .pkl format
model_filename = r'C:\miniproject\lstm_wind_model.pkl'
joblib.dump(best_model, model_filename)
print(f"✅ Model training completed and saved as '{model_filename}'!")
# Plot actual vs predicted values (wave graph)
plt.figure(figsize=(10, 5))
plt.plot(y_test_rescaled, label="Actual Wind Speed", linestyle='dashed', color='blue')
plt.plot(y_pred_rescaled, label="Predicted Wind Speed", linestyle='solid', color='red')
plt.xlabel("Time Steps")
plt.ylabel("Wind Speed at 100m (m/s)")
plt.title("Actual vs Predicted Wind Speed (Wave Graph)")
plt.legend()
plt.show()
