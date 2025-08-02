import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Load IMDB dataset
dataset, info = tfds.load("imdb_reviews", as_supervised=True, with_info=True)
train_data, test_data = dataset['train'], dataset['test']

# Convert dataset to numpy arrays
train_sentences, train_labels = [], []
test_sentences, test_labels = [], []

for sentence, label in train_data:
    train_sentences.append(sentence.numpy().decode('utf-8'))
    train_labels.append(label.numpy())

for sentence, label in test_data:
    test_sentences.append(sentence.numpy().decode('utf-8'))
    test_labels.append(label.numpy())

# Convert binary labels to 3-class labels (Manually creating a neutral class)
np.random.seed(42)
neutral_indices = np.random.choice(len(train_labels), size=int(0.3 * len(train_labels)), replace=False)

for i in neutral_indices:
    train_labels[i] = 1  # Assign Neutral class

# Convert labels to categorical (one-hot encoding)
train_labels = to_categorical(train_labels, num_classes=3)
test_labels = to_categorical(test_labels, num_classes=3)

# Tokenization and padding
vocab_size = 10000
max_length = 200
oov_token = "<OOV>"
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(train_sentences)

train_sequences = tokenizer.texts_to_sequences(train_sentences)
test_sequences = tokenizer.texts_to_sequences(test_sentences)

train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='post', truncating='post')
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post', truncating='post')

# Build LSTM Model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 64, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(3, activation='softmax')  # 3 Neurons for 3 classes
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
num_epochs = 5
history = model.fit(train_padded, train_labels, epochs=num_epochs, validation_data=(test_padded, test_labels), verbose=2)

# Evaluate model only on test dataset
test_loss, test_acc = model.evaluate(test_padded, test_labels)
print(f'\nTest Accuracy: {test_acc:.4f}')

# Plot accuracy and loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy over Epochs')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss over Epochs')
plt.show()

# Define reverse label mapping
label_mapping_reverse = {0: "Negative", 1: "Neutral", 2: "Positive"}

# Select a few sample test reviews
sample_reviews = test_sentences[:5]  # First 5 test reviews for demonstration

# Convert sample reviews into sequences
sample_sequences = tokenizer.texts_to_sequences(sample_reviews)
sample_padded = pad_sequences(sample_sequences, maxlen=max_length, padding='post')

# Predict sentiment
predictions = model.predict(sample_padded)
predicted_labels = np.argmax(predictions, axis=1)  # Get highest probability class

# Display results
print("\nSentiment Predictions:")
for review, sentiment in zip(sample_reviews, predicted_labels):
    print(f"Review: {review[:100]}... -> Sentiment: {label_mapping_reverse[sentiment]}")