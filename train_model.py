from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
import pickle

# Sample training data: you can expand this list
texts = [
    "Win money now!", 
    "Congratulations, you’ve been selected!",
    "You won a free iPhone!",
    "Meeting at 10am",
    "Can you send the report?",
    "Lunch at 2?",
    "Hey, are we still on for the meeting?",
    "Submit your assignment before 5pm"
]

# Labels: 1 = Spam, 0 = Not Spam
labels = [1, 1, 1, 0, 0, 0, 0, 0]

# Create a pipeline that includes vectorization and classification
pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', DecisionTreeClassifier())
])

# Train the model
pipeline.fit(texts, labels)

# Save the full pipeline as model.pkl
with open("model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("✅ Model trained and saved to model.pkl")
