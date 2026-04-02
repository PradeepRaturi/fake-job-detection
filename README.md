# fake-job-detection
AI-based system to identify fraudulent job postings using NLP techniques.
# 🧠 Fake Job Detection System

A Machine Learning project that detects whether a job posting is **real or fake** using Natural Language Processing (NLP) techniques.

---

## 📌 Overview

With the increasing number of online job postings, fraudulent listings have become a serious problem.
This project aims to automatically identify **fake job postings** and help users avoid scams.

---

## 🚀 Features

* 🔍 Classifies job postings as **Real** or **Fake**
* 🧠 Uses NLP (TF-IDF) for text processing
* ⚡ Fast and efficient Logistic Regression model
* 📊 Achieves high accuracy on real-world dataset

---

## 🛠️ Tech Stack

* **Python**
* **Pandas & NumPy**
* **Scikit-learn**
* **TF-IDF Vectorization**

---

## 📂 Dataset

* Source: Kaggle Fake Job Postings Dataset
* Contains both real and fraudulent job listings
* Key fields used:

  * `title`
  * `description`
  * `fraudulent`

---

## ⚙️ How It Works

1. Data Cleaning (handling missing values)
2. Text Preprocessing (combining title + description)
3. Feature Extraction using TF-IDF
4. Model Training using Logistic Regression
5. Prediction of job authenticity

---

## ▶️ How to Run

1. Install dependencies:

   ```
   pip install pandas numpy scikit-learn
   ```
2. Place dataset (`fake_job_postings.csv`) in project folder
3. Run:

   ```
   python model.py
   ```

---

## 🧪 Example

**Input:**
"Work from home, earn ₹50,000 daily with no experience required"

**Output:**
❌ Fake Job

---

## 📈 Future Improvements

* 🔮 Use advanced models (LSTM, BERT)
* 🌐 Build a web app (Streamlit)
* 📊 Improve accuracy with more features

---

## 👨‍💻 Author

 PRADEEP RATURI.

---

## ⭐ Contribution

Contributions are welcome! Feel free to fork this repo and improve the project.

---

## 📜 License

This project is for educational purposes only.
