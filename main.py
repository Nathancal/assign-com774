import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the WISDM dataset (replace 'path_to_data' with the actual path to your WISDM dataset)
path_to_data = 'WISDM_ar_v1.1/'
columns = ['user', 'activity', 'timestamp', 'x-axis', 'y-axis', 'z-axis']
df = pd.read_csv(path_to_data + 'WISDM_ar_v1.1_raw.txt', header=None, names=columns, comment=';')

df = df.dropna()

# Preprocess the data
print(df['z-axis'])

df['z-axis'] = df['z-axis'].astype(float)

# Extract features and labels
X = df[['x-axis', 'y-axis', 'z-axis']]
y = df['activity']

X_subset, _, y_subset, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_subset, y_subset, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=1000, random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))