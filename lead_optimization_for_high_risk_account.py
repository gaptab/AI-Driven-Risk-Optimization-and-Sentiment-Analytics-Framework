# Importing required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
# Assuming pre-labeled data for training
data = pd.DataFrame({
    'Account_ID': range(1, 1001),
    'Age': np.random.randint(18, 70, 1000),
    'Amount_Due': np.random.randint(500, 5000, 1000),
    'Delinquent_Payments': np.random.randint(0, 10, 1000),
    'Risk_Score': np.random.uniform(0, 1, 1000),  # Assume this is labeled for training
})

# Splitting data
X = data[['Age', 'Amount_Due', 'Delinquent_Payments']]
y = (data['Risk_Score'] > 0.7).astype(int)  # Binary classification

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predicting high-risk accounts
data['Predicted_Risk'] = model.predict(X)
data['Priority'] = np.where(data['Predicted_Risk'] == 1, 'High Risk', 'Low Risk')

high_risk_accounts = data[data['Priority'] == 'High Risk'].sort_values(by='Predicted_Risk', ascending=False)
print(high_risk_accounts)

data.to_csv("high_risk_accounts.csv", index=False)


# Dummy data for collection performance
performance_data = pd.DataFrame({
    'Week': ['Week 1', 'Week 2', 'Week 3', 'Week 4'],
    'Total_Collection': [120000, 135000, 128000, 140000],
    'High_Risk_Collected': [30000, 32000, 31000, 35000]
})
print(performance_data)

# Plotting the dashboard
plt.figure(figsize=(10, 5))
plt.plot(performance_data['Week'], performance_data['Total_Collection'], label='Total Collection', marker='o')
plt.plot(performance_data['Week'], performance_data['High_Risk_Collected'], label='High Risk Collected', marker='o')
plt.title('Collection Performance')
plt.xlabel('Week')
plt.ylabel('Amount Collected')
plt.legend()
plt.grid()
plt.show()

# Dummy SVM tagging
texts = ['Brand A launches new phone', 'Negative feedback about Brand B']
categories = ['Brand A', 'Brand B']
print(categories)

# Example data
texts = ['Brand A launches new phone', 'Negative feedback about Brand B']
labels = ['Brand A', 'Brand B']

# Vectorizing text
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Training SVM
svm = SVC()
svm.fit(X, labels)

# Classifying new text
new_texts = ['Positive review for Brand A', 'Issue reported with Brand B']
X_new = vectorizer.transform(new_texts)
predictions = svm.predict(X_new)
print(predictions)


