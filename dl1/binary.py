import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense

# Step 1: Load and preprocess the IMDB dataset
max_features = 10000  # Consider the top 10,000 most common words
maxlen = 200  # Cut off reviews after 200 words

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

# Step 2: Define the deep neural network model
model = Sequential([
    Embedding(max_features, 32, input_length=maxlen),
    Flatten(),
    Dense(1, activation='sigmoid')
])

# Step 3: Compile the model
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# Step 4: Train the model
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Step 5: Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print("Accuracy on Test Data:", accuracy)

# Step 6: Plot appropriate graphs
def plot_history(history):
    plt.plot(history.history['accuracy'], label='Accuracy (training data)')
    plt.plot(history.history['val_accuracy'], label='Accuracy (validation data)')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc="lower right")
    plt.show()

    plt.plot(history.history['loss'], label='Loss (training data)')
    plt.plot(history.history['val_loss'], label='Loss (validation data)')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc="upper right")
    plt.show()

plot_history(history)

# Step 7: Design a function to check your own review
def predict_review(review_text):
    # Tokenize the review text and pad/truncate it to maxlen
    review_sequence = pad_sequences([[word_index[word] if word in word_index and word_index[word] < max_features else 0 for word in review_text.split()]], maxlen=maxlen)
    # Predict sentiment (positive or negative)
    prediction = model.predict(review_sequence)[0][0]
    return "Positive" if prediction >= 0.5 else "Negative"

# Test the function with your own review
your_review = "I absolutely loved this movie! The acting was superb and the plot kept me engaged throughout."
print("Your review is:", predict_review(your_review))