import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


print("Tools loaded. Loading data...")

# LOAD THE DATA
df = pd.read_csv('spam.csv', encoding='latin-1')

df = df[['v1', 'v2']]
df.columns = ['label', 'message']

df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# SPLIT THE DATA (80-20 Split)

message_study, message_test, label_study, label_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)

# CLEAN THE TEXT AND TURN IT INTO NUMBERS
vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)

study_numbers = vectorizer.fit_transform(message_study)
test_numbers = vectorizer.transform(message_test)

# TRAIN THE MODEL
print("Training the AI...")
model = MultinomialNB()
model.fit(study_numbers, label_study)

# GIVE THE TEST AND GRADE IT
print("Taking the test...")
guesses = model.predict(test_numbers)

grade = accuracy_score(label_test, guesses)
print(f"Project Complete! The AI got a score of: {grade * 100:.2f}%")
