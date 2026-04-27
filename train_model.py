import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("loan_data.csv")

# Convert target
df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

# Handle missing values
df.fillna(method='ffill', inplace=True)

# Encode categorical columns
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# Split data
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict
preds = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, preds))
