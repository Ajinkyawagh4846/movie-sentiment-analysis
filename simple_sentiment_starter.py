"""
SENTIMENT ANALYSIS - SIMPLE STARTER VERSION
============================================
This single file does everything you need to understand the project.
We'll run this first, then expand to the full project.

What this does:
1. Loads a small dataset
2. Cleans the text
3. Trains a model
4. Makes predictions

Author: Your Name
Date: Today
"""

# ========== STEP 1: IMPORT LIBRARIES ==========
print("=" * 60)
print("STEP 1: Importing libraries...")
print("=" * 60)

import pandas as pd
import numpy as np
import re

# For text processing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

print("✓ All libraries imported successfully!\n")


# ========== STEP 2: CREATE SAMPLE DATA ==========
print("=" * 60)
print("STEP 2: Creating sample movie reviews...")
print("=" * 60)

# These are our training examples
reviews = [
    # Positive reviews (label = 1)
    "This movie was absolutely fantastic! I loved every minute of it.",
    "Amazing film with great acting and wonderful story.",
    "Best movie I've seen this year. Highly recommend!",
    "Brilliant performances and stunning visuals throughout.",
    "Excellent movie! Worth watching multiple times.",
    "Incredible story that kept me engaged from start to finish.",
    "Superb direction and outstanding cast. A masterpiece!",
    "Loved it! One of the best movies ever made.",
    "Perfect in every way. Cannot stop thinking about it.",
    "Awesome film with amazing plot twists.",
    
    # Negative reviews (label = 0)
    "Terrible movie. Complete waste of time and money.",
    "Worst film I've ever seen. Boring and predictable.",
    "Awful acting and a confusing plot. Do not watch.",
    "Horrible movie with no redeeming qualities whatsoever.",
    "Bad story, bad acting, bad everything. Disappointed.",
    "Dreadful film that made no sense at all.",
    "Poor execution and weak storyline. Very disappointing.",
    "Terrible waste of two hours. Would not recommend.",
    "Boring from start to finish. Fell asleep halfway.",
    "Disappointing movie with awful dialogue and acting."
] * 50  # Multiply to get 1000 reviews (500 positive, 500 negative)

# Labels: 1 = Positive, 0 = Negative
labels = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # First 10 are positive
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0] * 50  # Next 10 are negative

# Create a DataFrame (like an Excel table)
df = pd.DataFrame({
    'review': reviews,
    'sentiment': labels
})

print(f"✓ Created {len(df)} movie reviews")
print(f"  - Positive reviews: {sum(df['sentiment'] == 1)}")
print(f"  - Negative reviews: {sum(df['sentiment'] == 0)}")
print("\nFirst 3 reviews:")
print(df.head(3))
print()


# ========== STEP 3: CLEAN THE TEXT ==========
print("=" * 60)
print("STEP 3: Cleaning the text...")
print("=" * 60)

def clean_text(text):
    """
    This function cleans the review text:
    - Makes everything lowercase
    - Removes special characters
    - Keeps only letters and spaces
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters, keep only letters and spaces
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Example of cleaning
original = df['review'].iloc[0]
cleaned = clean_text(original)

print("BEFORE cleaning:")
print(f"  {original}")
print("\nAFTER cleaning:")
print(f"  {cleaned}")

# Clean all reviews
df['cleaned_review'] = df['review'].apply(clean_text)
print(f"\n✓ Cleaned all {len(df)} reviews\n")


# ========== STEP 4: SPLIT DATA ==========
print("=" * 60)
print("STEP 4: Splitting data into training and testing...")
print("=" * 60)

# We'll use 80% for training and 20% for testing
X = df['cleaned_review']  # The reviews
y = df['sentiment']        # The labels (1 or 0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"✓ Training set: {len(X_train)} reviews")
print(f"✓ Test set: {len(X_test)} reviews\n")


# ========== STEP 5: CONVERT TEXT TO NUMBERS ==========
print("=" * 60)
print("STEP 5: Converting text to numbers (TF-IDF)...")
print("=" * 60)

# Machine learning needs numbers, not text!
# TF-IDF converts text into numerical features
vectorizer = TfidfVectorizer(max_features=100)

# Learn from training data and transform
X_train_numbers = vectorizer.fit_transform(X_train)
X_test_numbers = vectorizer.transform(X_test)

print(f"✓ Converted text to {X_train_numbers.shape[1]} numerical features")
print(f"  Example words learned: {list(vectorizer.get_feature_names_out())[:10]}\n")


# ========== STEP 6: TRAIN THE MODEL ==========
print("=" * 60)
print("STEP 6: Training the AI model...")
print("=" * 60)

# Create and train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_numbers, y_train)

print("✓ Model training complete!\n")


# ========== STEP 7: TEST THE MODEL ==========
print("=" * 60)
print("STEP 7: Testing the model...")
print("=" * 60)

# Make predictions on test data
y_pred = model.predict(X_test_numbers)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"✓ Model Accuracy: {accuracy * 100:.2f}%\n")
print("Detailed Results:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))


# ========== STEP 8: TRY YOUR OWN REVIEWS ==========
print("\n" + "=" * 60)
print("STEP 8: Testing with custom reviews...")
print("=" * 60)

def predict_sentiment(review_text):
    """
    Predict if a review is positive or negative
    """
    # Clean the text
    cleaned = clean_text(review_text)
    
    # Convert to numbers
    vectorized = vectorizer.transform([cleaned])
    
    # Predict
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0]
    
    # Get results
    sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"
    confidence = probability[prediction] * 100
    
    return sentiment, confidence

# Test with custom reviews
test_reviews = [
    "I loved this movie so much!",
    "Boring and stupid film.",
    "Your review here!"
]

for review in test_reviews:
    sentiment, confidence = predict_sentiment(review)
    print(f"\nReview: {review}")
    print(f"→ Prediction: {sentiment}")
    print(f"→ Confidence: {confidence:.1f}%")


# ========== FINAL STEP: INTERACTIVE MODE ==========
print("\n" + "=" * 60)
print("🎉 PROJECT COMPLETE!")
print("=" * 60)
print("\nYou can now test with your own reviews!")
print("Copy this code and modify the test_reviews list above.")
print("\nTo run again: python simple_sentiment_starter.py")
print("=" * 60)


# ========== BONUS: SAVE THE MODEL ==========
print("\n💾 Saving the model for future use...")
import joblib

joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("✓ Model saved as: sentiment_model.pkl")
print("✓ Vectorizer saved as: vectorizer.pkl")
print("\nYou can load these later without retraining!")
