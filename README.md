# 🎬 IMDB Movie Review Sentiment Classifier

A deep learning web app that classifies movie reviews as **Positive** or **Negative** using a Simple RNN trained on the IMDB dataset. Built with TensorFlow/Keras and deployed via Streamlit.

---

## 📌 Project Overview

This project walks through the full ML pipeline — from data preprocessing and model training to real-time inference via a web interface.

- **Model:** Simple RNN with Embedding layer
- **Dataset:** IMDB (50,000 movie reviews — 25k train, 25k test)
- **Task:** Binary sentiment classification
- **Deployment:** Streamlit web app

---

## 🗂️ Project Structure

```
├── IMDB_review_classifier.ipynb   # Model training notebook
├── prediction.ipynb               # Inference/testing notebook
├── main.py                        # Streamlit web app
├── Simple_RNN_IMDB.keras          # Saved trained model
└── README.md
```

---

## 🧠 Model Architecture

| Layer | Output Shape | Parameters |
|-------|-------------|------------|
| Embedding | (None, 500, 128) | 1,280,000 |
| SimpleRNN | (None, 128) | 32,896 |
| Dense (sigmoid) | (None, 1) | 129 |

- **Total parameters:** ~1.3M
- **Optimizer:** Adam
- **Loss:** Binary Crossentropy
- **Early Stopping:** Monitors `val_loss` with patience of 5

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/imdb-sentiment-classifier.git
cd imdb-sentiment-classifier
```

### 2. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux
```

### 3. Install dependencies
```bash
pip install tensorflow streamlit numpy
```

---

## 🚀 Running the App

```bash
streamlit run main.py
```

Then open your browser at `http://localhost:8501`, type in any movie review, and hit **Classify**.

---

## 📓 Notebooks

| Notebook | Description |
|----------|-------------|
| `IMDB_review_classifier.ipynb` | Loads IMDB data, pads sequences, builds and trains the RNN, saves the model |
| `prediction.ipynb` | Loads the saved model, preprocesses raw text, and runs predictions |

---

## 🔧 How It Works

1. Raw review text is lowercased and split into words
2. Each word is mapped to its IMDB word index (vocabulary of 10,000 words)
3. The sequence is padded/truncated to a fixed length of 500
4. The model outputs a score between 0 and 1 — above 0.5 is **Positive**, below is **Negative**

---

## 📈 Training Results

The model was trained for 10 epochs with a validation split of 20%.

- **Best training accuracy:** ~97%
- **Best validation accuracy:** ~78%

> Note: There is some overfitting. Adding a `Dropout(0.5)` layer after the RNN and using `tanh` activation (instead of `relu`) is recommended for better generalization.

---

## 🛠️ Known Issues & Improvements

- `relu` activation in SimpleRNN can cause gradient explosion — use `tanh` instead
- Adding `Dropout` will reduce overfitting
- Upgrading to an LSTM or GRU would significantly improve accuracy

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
