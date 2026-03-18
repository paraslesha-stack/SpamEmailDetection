import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
df = pd.read_csv("data/spam.csv", encoding='latin-1')

# Keep only required columns
df = df[['v1', 'v2']]

# Rename columns
df.columns = ["label", "message"]

# Convert labels
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df["message"], df["label"], test_size=0.2, random_state=42
)

# Vectorize text
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Test
X_test_vec = vectorizer.transform(X_test)
accuracy = model.score(X_test_vec, y_test)

print("Model Accuracy:", accuracy)

# Test sample
sample = ["You have won 1 lakh rupees! Click now"]
sample_vec = vectorizer.transform(sample)
prediction = model.predict(sample_vec)

if prediction[0] == 1:
    print("This is SPAM")
else:
    print("This is NOT SPAM")