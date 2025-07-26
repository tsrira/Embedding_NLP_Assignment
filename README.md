# 🧠 Text Embedding Techniques in NLP

This project demonstrates and compares multiple text embedding techniques on a sample dataset using Python and libraries like `scikit-learn`, `nltk`, `gensim`, and `matplotlib`. The goal is to understand and analyze how different methods convert text into numerical vectors suitable for machine learning tasks.

---

## 📂 Contents

- 📌 Data Preparation & Preprocessing
- 🧮 Bag of Words (BoW)
- 🔠 TF-IDF
- 🧭 Word2Vec
- 🌍 GloVe (Pre-trained)
- 🧬 FastText
- 📊 Visualizations
- 📈 Comparative Summary

---

## 📥 GloVe Download (Required for Step 5)

To use GloVe embeddings:

1. Download from: https://nlp.stanford.edu/projects/glove/
2. Choose: `glove.6B.zip`
3. Unzip and place the file `glove.6B.100d.txt` in your project directory.

### 📌 Load GloVe into a Dictionary

```python
glove_embeddings = {}
with open("glove.6B.100d.txt", 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        glove_embeddings[word] = vector
```

---

## 🔧 Step-by-Step Implementation

### 1️⃣ Setup & Preprocessing

- Load the IMDb movie review dataset from NLTK
- Clean the text: lowercase, remove digits, punctuation, stop words
- Tokenize and prepare text for modeling

---

### 2️⃣ Bag of Words (BoW)

- Used `CountVectorizer` from `sklearn`
- Creates a sparse matrix of word counts
- Simple but ignores word meaning and context

---

### 3️⃣ TF-IDF

- Used `TfidfVectorizer`
- Weighs important words higher and downweights common words
- Improves over BoW for most NLP tasks

---

### 4️⃣ Word2Vec

- Trained using Gensim’s `Word2Vec` with skip-gram
- Learns word embeddings based on local context
- Words with similar meanings lie close in vector space

---

### 5️⃣ GloVe (Global Vectors)

- Used pre-trained `glove.6B.100d.txt`
- Captures global co-occurrence statistics
- Useful when training data is limited

---

### 6️⃣ FastText

- Trained using Gensim’s `FastText`
- Uses subword information (character n-grams)
- Supports out-of-vocabulary (OOV) words and rare words

---

## 📊 Embedding Technique Comparison

| Feature                  | BoW                     | TF-IDF                 | Word2Vec               | GloVe                   | FastText                   |
|--------------------------|--------------------------|-------------------------|-------------------------|--------------------------|-----------------------------|
| Type                     | Sparse count matrix      | Weighted sparse matrix  | Dense vectors           | Dense vectors            | Dense vectors + subwords    |
| Context Captured         | ❌                       | ❌                      | ✅ (local)              | ✅ (global)              | ✅ (local + subword)         |
| OOV Word Handling        | ❌                       | ❌                      | ❌                      | ❌                       | ✅                          |
| Pretrained Available     | ❌                       | ❌                      | ✅                      | ✅                       | ✅                          |
| Morphological Awareness  | ❌                       | ❌                      | ❌                      | ❌                       | ✅                          |
| Interpretability         | ✅                       | ✅                      | ❌                      | ❌                       | ❌                          |

---

## 📈 Visualizations

- Bar plots for most frequent terms
- PCA-based scatter plots for Word2Vec, GloVe, and FastText embeddings

---

## 📌 How to Run

1. Ensure you have the following Python packages installed:
```bash
pip install pandas numpy nltk gensim scikit-learn matplotlib seaborn
```

2. Download and place GloVe file as described above.

3. Run the notebook: `Sriram_Embedding_Assignment.ipynb`

---

## 📄 License

This project is for academic and educational purposes.
