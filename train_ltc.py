import numpy as np
import pandas as pd
from tensorflow import keras
from ncps import wirings
from ncps.tf import LTC
import matplotlib.pyplot as plt
import seaborn as sns

# Load the stock data from CSV
data = pd.read_csv('GOOGL.csv')

# Normalize the data
data = data.drop(columns=['Date'])  # Remove Date column for modeling
data = (data - data.mean()) / data.std()  # Normalize data

# Prepare input-output sequences for the LSTM
sequence_length = 30  # Length of the time-series sequences
data_x, data_y = [], []

for i in range(len(data) - sequence_length):
    data_x.append(data.iloc[i:i+sequence_length].values)
    data_y.append(data.iloc[i+sequence_length]['Close'])  # Predicting 'Close' feature

data_x = np.array(data_x)
data_y = np.array(data_y)
print("data_x.shape:", data_x.shape)
print("data_y.shape:", data_y.shape)

# Design the LTC model
wiring = wirings.AutoNCP(16, 1)  # 16 neurons, 1 output (for predicting Close)
model = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(sequence_length, data.shape[1])),
    LTC(wiring, return_sequences=False),
])
model.compile(optimizer=keras.optimizers.legacy.Adam(0.01), loss='mean_squared_error')

model.summary()

# Visualize LTC initial wiring
sns.set_style("white")
plt.figure(figsize=(6, 4))
legend_handles = wiring.draw_graph(draw_labels=True, neuron_colors={"command": "tab:cyan"})
plt.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(1, 1))
sns.despine(left=True, bottom=True)
plt.tight_layout()
plt.savefig("./results/initial_wiring.png")

# Visualize before training
prediction = model(data_x).numpy()
plt.figure(figsize=(6, 4))
plt.plot(data_y, label="Target output")
plt.plot(prediction[:, 0], label="LTC Prediction", linestyle="dashed")
plt.title("Before training")
plt.legend(loc="upper left")
plt.savefig("./results/before_training.png")

# Train the model
hist = model.fit(x=data_x, y=data_y, batch_size=1, epochs=100, verbose=1)

# Visualize training loss
plt.figure(figsize=(6, 4))
plt.plot(hist.history["loss"], label="Training loss")
plt.legend(loc="upper right")
plt.xlabel("Training steps")
plt.savefig("./results/training_loss.png")

# Visualize after training
prediction = model(data_x).numpy()
plt.figure(figsize=(6, 4))
plt.plot(data_y, label="Target output")
plt.plot(prediction[:, 0], label="LTC Prediction", linestyle="dashed")
plt.legend(loc="upper left")
plt.title("After training")
plt.savefig("./results/after_training.png")

# Save the model in keras format
model.save("ltc_model.keras")
print("Model saved successfully.")

# predict next 30 days
data_next_seq = data.iloc[-sequence_length:].values
data_next_seq = np.expand_dims(data_next_seq, axis=0)

prediction = model.predict(data_next_seq)
print("Prediction for next 30 days:", prediction)