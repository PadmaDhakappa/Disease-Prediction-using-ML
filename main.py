import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib


# Load dataset
data = pd.read_csv('diabetes.csv')
data.fillna(data.mean(), inplace=True)

# Check for issues in the data
print(data.isnull().sum())
print(data.shape)

# Define features and target
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Ensure consistency
assert len(X) == len(y), "Mismatch in lengths of X and y"

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of the model is:", accuracy)

#Save the trained Model:
joblib.dump(model, 'diabetes_model.pkl')


hdata=pd.read_csv('heart.csv')
print(hdata.info())
hdata.fillna(hdata.mean(), inplace=True)

# Check for issues in the data
print(hdata.isnull().sum())
print(hdata.shape)

# Define features and target
X = hdata.drop('target', axis=1)
y = hdata['target']

# Ensure consistency
assert len(X) == len(y), "Mismatch in lengths of X and y"

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of the model is:", accuracy)

#Save the trained Model:
joblib.dump(model, 'heart_model.pkl')

pdata=pd.read_csv('parkinsons.csv')
print(pdata.info())

pdata = pdata.drop(['name'], axis=1)

# Check for issues in the data
print(pdata.isnull().sum())
print(pdata.shape)

# Define features and target
X = pdata.drop('status', axis=1)
y = pdata['status']

# Ensure consistency
assert len(X) == len(y), "Mismatch in lengths of X and y"

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of the model is:", accuracy)

#Save the trained Model:
joblib.dump(model, 'parkinsons_model.pkl')