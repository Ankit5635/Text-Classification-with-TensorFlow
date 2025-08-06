# Text-Classification-with-TensorFlow

# IMDB Movie Reviews Sentiment Classification (3-Class LSTM Model)

This project builds and trains a **Bidirectional LSTM** model to classify IMDB movie reviews into **three sentiment classes**: **Negative**, **Neutral**, and **Positive**. The dataset is loaded from TensorFlow Datasets, and the model is implemented using TensorFlow and Keras.

## 📌 Project Highlights

- ✅ Uses the IMDB Reviews dataset from TensorFlow Datasets.
- ✅ Converts binary sentiment labels (positive/negative) into 3 classes by randomly assigning a "Neutral" label to 30% of the training set.
- ✅ Preprocessing includes tokenization and padding.
- ✅ A Bidirectional LSTM model is trained to classify reviews.
- ✅ Evaluation includes accuracy, loss plots, and sample predictions.

---

## 🔧 Technologies Used

- Python
- TensorFlow 2.x
- TensorFlow Datasets (TFDS)
- NumPy
- Matplotlib

---

## 📂 Dataset

- **Source:** `imdb_reviews` from TensorFlow Datasets
- **Original Format:** Binary classification (positive/negative)
- **Modified Format:** Ternary classification (Negative, Neutral, Positive)

---

## 🧠 Model Architecture

```text
Embedding Layer (vocab_size=10000, output_dim=64)
↓
Bidirectional LSTM (64 units, return_sequences=True)
↓
Bidirectional LSTM (32 units)
↓
Dense Layer (64 units, ReLU)
↓
Dropout (rate=0.5)
↓
Dense Output Layer (3 units, Softmax)
