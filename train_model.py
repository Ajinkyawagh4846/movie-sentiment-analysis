"""
SENTIMENT ANALYSIS - SIMPLE VERSION WITH CSV
=============================================
This version reads data from a CSV file (much easier!)

Files needed:
1. This file: train_model.py
2. Data file: IMDB Dataset.csv (put both in same folder)

Author: Your Name
Date: Today
"""

# ========== STEP 1: IMPORT LIBRARIES ==========
print("=" * 70)
print("STEP 1: Importing libraries...")
print("=" * 70)

import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("✓ All libraries imported successfully!\n")


# ========== STEP 2: LOAD DATA FROM CSV ==========
print("=" * 70)
print("STEP 2: Loading data from CSV file...")
print("=" * 70)

try:
    # Load the CSV file (make sure it's in the same folder!)
    df = pd.read_csv('IMDB Dataset.csv')
    # Convert 'positive' -> 1 and 'negative' -> 0
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
 
    
    print(f"✓ Successfully loaded dataset!")
    print(f"  Total reviews: {len(df)}")
    print(f"  Positive reviews: {sum(df['sentiment'] == 1)}")
    print(f"  Negative reviews: {sum(df['sentiment'] == 0)}")
    
    # Show first few rows
    print("\n📋 First 3 reviews from dataset:")
    print("-" * 70)
    for i in range(3):
        sentiment_label = "POSITIVE ✅" if df['sentiment'].iloc[i] == 1 else "NEGATIVE ❌"
        print(f"{i+1}. [{sentiment_label}] {df['review'].iloc[i]}")
    print()
    
except FileNotFoundError:
    print("\n❌ ERROR: Could not find 'IMDB Dataset.csv")
    print("\nPlease make sure:")
    print("  1. The file 'IMDB Dataset.csv' is in the same folder")
    print("  2. You're running this script from the correct folder")
    print("\nCurrent folder files:")
    import os
    print("  " + "\n  ".join(os.listdir('.')))
    exit()


# ========== STEP 3: CLEAN THE TEXT ==========
print("=" * 70)
print("STEP 3: Cleaning all reviews...")
print("=" * 70)

def clean_text(text):
    """
    Clean the review text:
    - Lowercase
    - Remove special characters
    - Remove extra spaces
    """
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Show example
print("Example of cleaning:")
original = df['review'].iloc[0]
cleaned = clean_text(original)
print(f"Before: {original}")
print(f"After:  {cleaned}")

# Clean all reviews
df['cleaned_review'] = df['review'].apply(clean_text)
print(f"\n✓ Cleaned all {len(df)} reviews\n")


# ========== STEP 4: SPLIT DATA ==========
print("=" * 70)
print("STEP 4: Splitting data into training and testing...")
print("=" * 70)

X = df['cleaned_review']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ Training set: {len(X_train)} reviews (80%)")
print(f"✓ Test set: {len(X_test)} reviews (20%)\n")


# ========== STEP 5: CONVERT TEXT TO NUMBERS ==========
print("=" * 70)
print("STEP 5: Converting text to numbers (TF-IDF)...")
print("=" * 70)

vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"✓ Created {X_train_tfidf.shape[1]} features")
print(f"  Top words: {list(vectorizer.get_feature_names_out())[:15]}\n")


# ========== STEP 6: TRAIN THE MODEL ==========
print("=" * 70)
print("STEP 6: Training the AI model...")
print("=" * 70)

model = LogisticRegression(max_iter=1000, random_state=42)
print("Training in progress...")
model.fit(X_train_tfidf, y_train)

print("✓ Model training complete!\n")


# ========== STEP 7: EVALUATE THE MODEL ==========
print("=" * 70)
print("STEP 7: Testing model accuracy...")
print("=" * 70)

y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n🎯 MODEL ACCURACY: {accuracy * 100:.2f}%\n")
print("Detailed Performance Report:")
print("-" * 70)
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))


# ========== STEP 8: OVERALL STATISTICS ==========
print("\n" + "=" * 70)
print("STEP 8: ANALYZING ALL REVIEWS - STATISTICS YOU REQUESTED!")
print("=" * 70)

# Predict on ALL reviews
all_reviews_tfidf = vectorizer.transform(df['cleaned_review'])
all_predictions = model.predict(all_reviews_tfidf)

# Calculate statistics
total_reviews = len(all_predictions)
positive_count = sum(all_predictions == 1)
negative_count = sum(all_predictions == 0)
positive_percent = (positive_count / total_reviews) * 100
negative_percent = (negative_count / total_reviews) * 100

# Display results - EXACTLY WHAT YOU ASKED FOR!
print("\n" + "🎬" * 35)
print("📊 COMPLETE DATASET ANALYSIS")
print("🎬" * 35)

print(f"\n📝 Total Reviews Analyzed: {total_reviews:,}")
print("=" * 70)

print(f"\n✅ POSITIVE Reviews: {positive_count:,} reviews")
print(f"   Percentage: {positive_percent:.1f}%")

print(f"\n❌ NEGATIVE Reviews: {negative_count:,} reviews")  
print(f"   Percentage: {negative_percent:.1f}%")

print("\n" + "=" * 70)

# Visual bar chart
print("\n📊 VISUAL DISTRIBUTION:")
print("-" * 70)
bar_length = 50
pos_bars = int((positive_percent / 100) * bar_length)
neg_bars = int((negative_percent / 100) * bar_length)

print(f"Positive: {'🟢' * pos_bars}{'⬜' * (bar_length - pos_bars)} {positive_percent:.1f}%")
print(f"Negative: {'🔴' * neg_bars}{'⬜' * (bar_length - neg_bars)} {negative_percent:.1f}%")
print("-" * 70)

# Summary box
print(f"""
╔════════════════════════════════════════════════════════════════════╗
║                        SUMMARY STATISTICS                          ║
╠════════════════════════════════════════════════════════════════════╣
║  Total Reviews:       {total_reviews:,}                            ║
║  Positive Reviews:    {positive_count:,} ({positive_percent:.1f}%) ║
║  Negative Reviews:    {negative_count:,} ({negative_percent:.1f}%) ║
║  Model Accuracy:      {accuracy * 100:.1f}%                        ║
╚════════════════════════════════════════════════════════════════════╝
""")


# ========== STEP 9: TEST YOUR OWN REVIEWS ==========
print("\n" + "=" * 70)
print("STEP 9: Test with your own movie reviews!")
print("=" * 70)

def predict_sentiment(review_text):
    cleaned = clean_text(review_text)
    vectorized = vectorizer.transform([cleaned])
    prediction = int(model.predict(vectorized)[0])
    probability = model.predict_proba(vectorized)[0]
    
    sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"
    confidence = probability[prediction] * 100
    
    return sentiment, confidence



# Test with sample reviews
test_reviews = [
    "This movie was absolutely amazing! Best film ever!",
    "Terrible waste of time. Boring and poorly made.",
    "It was decent, nothing special but entertaining."
]

print("\n📝 Sample Predictions:")
print("-" * 70)

for i, review in enumerate(test_reviews, 1):
    sentiment, confidence = predict_sentiment(review)
    print(f"\n{i}. Review: \"{review}\"")
    print(f"   → Prediction: {sentiment}")
    print(f"   → Confidence: {confidence:.1f}%")

print("\n" + "-" * 70)


# ========== STEP 10: SAVE MODEL ==========
print("\n" + "=" * 70)
print("STEP 10: Saving the trained model...")
print("=" * 70)

joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("✓ Model saved as: sentiment_model.pkl")
print("✓ Vectorizer saved as: vectorizer.pkl")
print("\n💡 You can load these files later without retraining!")


# ========== FINAL MESSAGE ==========
print("\n" + "🎉" * 35)
print("PROJECT COMPLETE - GREAT JOB!")
print("🎉" * 35)

print(f"""
📊 FINAL SUMMARY:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Analyzed {total_reviews:,} movie reviews from CSV file
✓ Model Accuracy: {accuracy * 100:.1f}%
✓ Found {positive_count:,} Positive reviews ({positive_percent:.1f}%)
✓ Found {negative_count:,} Negative reviews ({negative_percent:.1f}%)
✓ Model saved successfully!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🚀 WHAT'S NEXT?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Edit the 'test_reviews' list above with YOUR reviews
2. Run the script again: python train_model.py
3. Your model is saved - no need to retrain!
4. Ready to build a web app? Just ask!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

print("=" * 70)
print("🎓 Congratulations! You're now a sentiment analysis expert!")
print("=" * 70)
