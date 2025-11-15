import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv("data.csv", sep=';')

# Clean column names
df.columns = df.columns.str.strip().str.replace('"', '').str.replace('\t', '')

# Features and target
X = df.drop('Target', axis=1)
y = df['Target']

# Encode y
le = LabelEncoder()
y = le.fit_transform(y)
joblib.dump(le, 'label_encoder.pkl')

# Scale X
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'scaler.pkl')

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'model.pkl')

# Evaluate
y_pred = model.predict(X_test)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred)}")