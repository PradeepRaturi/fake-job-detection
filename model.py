# Step 1: Import libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 2: Load dataset
df = pd.read_csv("fake_job_postings.csv")

# Step 3: Select useful columns
df = df[['title', 'description', 'fraudulent']]

# Step 4: Handle missing values
df = df.fillna('')

# Step 5: Combine text data
df['text'] = df['title'] + " " + df['description']

# Step 6: Split data
X = df['text']
y = df['fraudulent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Step 7: Convert text to numbers
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Step 8: Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 9: Prediction
y_pred = model.predict(X_test)

# Step 10: Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Step 11: Test custom input
def predict_job(text):
    input_data = vectorizer.transform([text])
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        print("❌ Fake Job")
    else:
        print("✅ Real Job")

# Example test
predict_job("Work from home, earn ₹50000 daily without any skills")
