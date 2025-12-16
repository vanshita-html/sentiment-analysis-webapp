import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import joblib

# 1. Load the dataset
print("Loading dataset...")
data = pd.read_csv('data.csv')

# 2. Split data into training and testing sets
# We hide a small part of data (20%) to test the model later
X = data['text'] # The features (text)
y = data['sentiment'] # The target (labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Build the Pipeline
# A "Pipeline" chains steps together:
# Step A: CountVectorizer turns text into numbers (mathematical vectors)
# Step B: MultinomialNB is the algorithm (Naive Bayes) that learns from those numbers
model = make_pipeline(CountVectorizer(), MultinomialNB())

# 4. Train the model
print("Training the model...")
model.fit(X_train, y_train)

# 5. Evaluate the model
print("Evaluating model...")
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# 6. Save the trained model to a file
joblib.dump(model, 'sentiment_model.pkl')
print("Model saved as 'sentiment_model.pkl'")