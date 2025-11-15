# generate_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Load and clean data
df = pd.read_csv("data.csv", sep=';')
df.columns = df.columns.str.strip().str.replace('"', '').str.replace('\t', ' ')
df.dropna(inplace=True)

X = df.drop('Target', axis=1)
y = df['Target']

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)
joblib.dump(le, 'label_encoder.pkl')

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'scaler.pkl')

# Train model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
joblib.dump(model, 'model.pkl')

print("All 3 files generated: model.pkl, scaler.pkl, label_encoder.pkl")
print(f"Accuracy: {model.score(X_test, y_test):.4f}")