import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Bidirectional, Input
print("✅ Before AdamW import")
from tensorflow.keras.optimizers import AdamW
print("✅ After AdamW import")


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv(r'C:\miniproject\preprocessed_wind_data.csv')

# Select features and target
features = ['wind direction at 100m (deg)', 'Timestamp']
target = 'wind speed at 100m (m/s)'

# Scale features and target
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
df[features] = scaler_X.fit_transform(df[features])
df[target] = scaler_y.fit_transform(df[[target]])

# Sequence creation for LSTM
def create_sequences(data, target, sequence_length=50):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(target[i+sequence_length])
    return np.array(X), np.array(y)

sequence_length = 50
X, y = create_sequences(df[features].values, df[target].values, sequence_length)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Chaos Game Optimization
class CGO:
    def __init__(self, objective_function, bounds, num_generations=5, population_size=10):
        self.objective_function = objective_function
        self.bounds = bounds
        self.num_generations = num_generations
        self.population_size = population_size
        self.dim = len(bounds)

    def initialize_population(self):
        return np.array([[np.random.uniform(low, high) for low, high in self.bounds] for _ in range(self.population_size)])

    def evolve(self, population):
        new_population = []
        for i in range(len(population)):
            j = np.random.randint(0, len(population))
            new_individual = population[i] + np.random.uniform(-0.1, 0.1, self.dim) * (population[j] - population[i])
            new_individual = np.clip(new_individual, [b[0] for b in self.bounds], [b[1] for b in self.bounds])
            new_population.append(new_individual)
        return np.array(new_population)

    def run(self):
        population = self.initialize_population()
        best_solution = None
        best_score = float('inf')
        for gen in range(self.num_generations):
            scores = np.array([self.objective_function(ind) for ind in population])
            best_idx = np.argmin(scores)
            if scores[best_idx] < best_score:
                best_solution = population[best_idx]
                best_score = scores[best_idx]
            population = self.evolve(population)
        return best_solution

# Objective function using LSTM
def lstm_cgo_objective(params):
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        Bidirectional(LSTM(units=int(params[0]), return_sequences=True)),
        BatchNormalization(),
        Dropout(0.2),

        Bidirectional(LSTM(units=int(params[1]), return_sequences=True)),
        BatchNormalization(),
        Dropout(0.2),

        Bidirectional(LSTM(units=int(params[2]))),
        BatchNormalization(),
        Dropout(0.2),

        Dense(1)
    ])
    
    model.compile(optimizer=AdamW(learning_rate=params[3]), loss='mse', metrics=['mae'])
    model.fit(X_train, y_train, epochs=3, batch_size=32, verbose=0, validation_split=0.2)
    val_loss = model.evaluate(X_test, y_test, verbose=0)[0]
    return val_loss

# Parameter bounds: [LSTM1 units, LSTM2 units, LSTM3 units, Learning rate]
param_bounds = [
    (128, 256),  # LSTM Layer 1 units
    (128, 256),  # LSTM Layer 2 units
    (64, 128),   # LSTM Layer 3 units
    (0.0001, 0.001)  # Learning rate
]

# Run CGO
cgo = CGO(objective_function=lstm_cgo_objective, bounds=param_bounds, num_generations=3, population_size=5)
best_params = cgo.run()
print(f"Best CGO Parameters: {best_params}")

# Final model with best parameters
best_model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    Bidirectional(LSTM(units=int(best_params[0]), return_sequences=True)),
    BatchNormalization(),
    Dropout(0.2),

    Bidirectional(LSTM(units=int(best_params[1]), return_sequences=True)),
    BatchNormalization(),
    Dropout(0.2),

    Bidirectional(LSTM(units=int(best_params[2]))),
    BatchNormalization(),
    Dropout(0.2),

    Dense(1)
])

best_model.compile(optimizer=AdamW(learning_rate=best_params[3]), loss='mse', metrics=['mae'])
history = best_model.fit(X_train, y_train, epochs=3, validation_data=(X_test, y_test), batch_size=32)

# Predictions and evaluation
y_pred = best_model.predict(X_test)
y_pred_rescaled = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
y_test_rescaled = scaler_y.inverse_transform(y_test.reshape(-1, 1))

mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
print(f"Optimized Test MAE: {mae:.4f} m/s")

# Save the model properly
model_path = r'C:\miniproject\lstm_wind_model_cgo.keras'
best_model.save(model_path)
print(f"✅ CGO-optimized model saved as '{model_path}'!")

# Plot predictions
plt.figure(figsize=(12, 6))
plt.plot(y_test_rescaled, label="Actual Wind Speed", linestyle='dashed', color='blue')
plt.plot(y_pred_rescaled, label="Predicted Wind Speed", linestyle='solid', color='red')
plt.xlabel("Time Steps")
plt.ylabel("Wind Speed at 100m (m/s)")
plt.title("CGO-Optimized Actual vs Predicted Wind Speed (Wave Graph)")
plt.legend()
plt.grid(True)
plt.show()
