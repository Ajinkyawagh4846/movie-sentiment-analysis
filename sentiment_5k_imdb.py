"""
SENTIMENT ANALYSIS - REAL IMDB DATA VERSION
============================================
This version uses REAL movie reviews from IMDB dataset (5,000 reviews)
Shows detailed statistics like you requested!

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
from tqdm import tqdm  # For progress bars

# For text processing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

print("✓ All libraries imported successfully!\n")


# ========== STEP 2: LOAD REAL IMDB DATA ==========
print("=" * 70)
print("STEP 2: Loading REAL IMDB movie reviews...")
print("=" * 70)

try:
    import tensorflow as tf
    print("Downloading IMDB dataset... (this may take 1-2 minutes first time)")
    
    # Load IMDB dataset
    (train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.imdb.load_data(num_words=10000)
    
    # Get word index to decode reviews
    word_index = tf.keras.datasets.imdb.get_word_index()
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    
    # Function to decode reviews
    def decode_review(encoded_review):
        return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])
    
    # Decode 5,000 reviews (2,500 from train, 2,500 from test)
    print("\nDecoding reviews... (this takes 1-2 minutes)")
    num_samples = 2500
    
    train_reviews = []
    for review in tqdm(train_data[:num_samples], desc="Train reviews"):
        train_reviews.append(decode_review(review))
    
    test_reviews = []
    for review in tqdm(test_data[:num_samples], desc="Test reviews"):
        test_reviews.append(decode_review(review))
    
    # Combine into one dataset
    all_reviews = train_reviews + test_reviews
    all_labels = list(train_labels[:num_samples]) + list(test_labels[:num_samples])
    
    # Create DataFrame
    df = pd.DataFrame({
        'review': all_reviews,
        'sentiment': all_labels
    })
    
    print(f"\n✓ Successfully loaded {len(df)} REAL IMDB reviews!")
    print(f"  - Positive reviews: {sum(df['sentiment'] == 1)}")
    print(f"  - Negative reviews: {sum(df['sentiment'] == 0)}")
    
except Exception as e:
    print(f"\n⚠️  Could not load IMDB dataset: {e}")
    print("Installing TensorFlow... please wait...")
    import subprocess
    subprocess.run(["pip", "install", "tensorflow", "--break-system-packages"], 
                   capture_output=True)
    print("\nPlease run the script again after TensorFlow is installed!")
    exit()

# Show sample review
print("\n" + "=" * 70)
print("📖 Sample Review (original):")
print("=" * 70)
print(df['review'].iloc[0][:300] + "...")
print()


# ========== STEP 3: CLEAN THE TEXT ==========
print("=" * 70)
print("STEP 3: Cleaning all reviews...")
print("=" * 70)

def clean_text(text):
    """Clean and preprocess text"""
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)  # Remove HTML
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-z\s]', '', text)  # Keep only letters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Clean all reviews with progress bar
print("Cleaning reviews...")
tqdm.pandas(desc="Progress")
df['cleaned_review'] = df['review'].progress_apply(clean_text)

print(f"\n✓ Cleaned all {len(df)} reviews")
print("\n📖 Sample Review (after cleaning):")
print(df['cleaned_review'].iloc[0][:300] + "...")
print()


# ========== STEP 4: SPLIT DATA ==========
print("=" * 70)
print("STEP 4: Splitting data...")
print("=" * 70)

X = df['cleaned_review']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ Training set: {len(X_train)} reviews")
print(f"✓ Test set: {len(X_test)} reviews\n")


# ========== STEP 5: CONVERT TEXT TO NUMBERS ==========
print("=" * 70)
print("STEP 5: Creating TF-IDF features...")
print("=" * 70)

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=5)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"✓ Created {X_train_tfidf.shape[1]} features")
print(f"  Sample features: {list(vectorizer.get_feature_names_out())[:15]}\n")


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


# ========== STEP 8: OVERALL STATISTICS (YOUR REQUEST!) ==========
print("\n" + "=" * 70)
print("STEP 8: ANALYZING ALL REVIEWS - OVERALL STATISTICS")
print("=" * 70)

# Predict on ALL reviews in the dataset
print("\nMaking predictions on all reviews...")
all_reviews_tfidf = vectorizer.transform(df['cleaned_review'])
all_predictions = model.predict(all_reviews_tfidf)

# Calculate statistics
total_reviews = len(all_predictions)
positive_count = sum(all_predictions == 1)
negative_count = sum(all_predictions == 0)
positive_percent = (positive_count / total_reviews) * 100
negative_percent = (negative_count / total_reviews) * 100

# Display results
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

# Additional insights
print("\n💡 INSIGHTS:")
if abs(positive_percent - 50) < 5:
    print("   ✓ Dataset is well-balanced (good for training!)")
elif positive_percent > negative_percent:
    print(f"   → More positive reviews ({positive_percent - negative_percent:.1f}% difference)")
else:
    print(f"   → More negative reviews ({negative_percent - positive_percent:.1f}% difference)")

print("\n" + "=" * 70)


# ========== STEP 9: TEST YOUR OWN REVIEWS ==========
print("\n" + "=" * 70)
print("STEP 9: Try your own movie reviews!")
print("=" * 70)

def predict_sentiment(review_text):
    """Predict sentiment for a new review"""
    cleaned = clean_text(review_text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0]
    
    sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"
    confidence = probability[prediction] * 100
    
    return sentiment, confidence

# Test with custom reviews
test_reviews = [
    "This movie was absolutely amazing! Best film I have ever seen!",
    "Terrible waste of time. Boring and poorly made.",
    "It was okay, nothing special but not terrible either."
]

print("\n📝 Testing Custom Reviews:")
print("-" * 70)

for i, review in enumerate(test_reviews, 1):
    sentiment, confidence = predict_sentiment(review)
    print(f"\n{i}. Review: \"{review}\"")
    print(f"   Prediction: {sentiment}")
    print(f"   Confidence: {confidence:.1f}%")

print("\n" + "-" * 70)


# ========== STEP 10: SAVE MODEL ==========
print("\n" + "=" * 70)
print("STEP 10: Saving the trained model...")
print("=" * 70)

import joblib

joblib.dump(model, 'sentiment_model_5k.pkl')
joblib.dump(vectorizer, 'vectorizer_5k.pkl')

print("✓ Model saved as: sentiment_model_5k.pkl")
print("✓ Vectorizer saved as: vectorizer_5k.pkl")
print("\nYou can load these files later without retraining!")


# ========== FINAL SUMMARY ==========
print("\n" + "🎉" * 35)
print("PROJECT COMPLETE!")
print("🎉" * 35)

print(f"""
📊 SUMMARY:
-----------
✓ Trained on {total_reviews:,} real IMDB reviews
✓ Model Accuracy: {accuracy * 100:.1f}%
✓ Positive Reviews: {positive_count:,} ({positive_percent:.1f}%)
✓ Negative Reviews: {negative_count:,} ({negative_percent:.1f}%)

💾 Models saved and ready to use!

🚀 NEXT STEPS:
--------------
1. Modify the test_reviews list above with YOUR own reviews
2. Run again: python sentiment_5k_imdb.py
3. Or load the model in a new script without retraining!

""")

print("=" * 70)
print("Great job! You just trained an AI model on real data! 🎓")
print("=" * 70)
