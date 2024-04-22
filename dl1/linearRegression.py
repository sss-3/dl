import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Step 1: Load and preprocess the Boston housing dataset
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

# Normalize the features
mean = x_train.mean(axis=0)
std = x_train.std(axis=0)
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

# Step 2: Define the deep neural network model for linear regression
model = Sequential([
    Dense(1, input_shape=(x_train.shape[1],))
])

# Step 3: Compile the model
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

# Step 4: Train the model
history = model.fit(x_train, y_train, epochs=100, batch_size=16, validation_split=0.2, verbose=0)

# Step 5: Evaluate the model
loss, mae = model.evaluate(x_test, y_test)
print("Mean Absolute Error on Test Data:", mae)

# Step 6: Plot appropriate graphs
def plot_history(history):
    plt.plot(history.history['mae'], label='Mean Absolute Error (training data)')
    plt.plot(history.history['val_mae'], label='Mean Absolute Error (validation data)')
    plt.title('Model Mean Absolute Error')
    plt.ylabel('Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.legend(loc="upper right")
    plt.show()

    plt.plot(history.history['loss'], label='Loss (training data)')
    plt.plot(history.history['val_loss'], label='Loss (validation data)')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc="upper right")
    plt.show()

plot_history(history)