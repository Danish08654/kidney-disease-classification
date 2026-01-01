import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Step 1: Load and preprocess data (same as during training)
df = pd.read_csv(r"F:\lft\Liver function test.csv")
df.drop(columns=['Patient ID'], inplace=True)
df['Gender'] = LabelEncoder().fit_transform(df['Gender'])

# Step 2: Split features and target
X = df.drop(columns=['Liver_Disease'])
y = df['Liver_Disease']

# Step 3: Split into training and test sets (test = 30%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Step 4: Load the saved model
model = joblib.load(r"F:\lft\liver_disease_adaboost_pipeline.pkl")

# Step 5: Predict on the test set
y_test_pred = model.predict(X_test)

# Step 6: Evaluate the model
print("Evaluation on Real Test Set:")
print(classification_report(y_test, y_test_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))
print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
