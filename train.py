import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics

data = pd.read_csv('data/iris.csv')

train, test = train_test_split(data, test_size = 0.4, stratify = data['species'], random_state = 42)
X_train = train[['sepal_length','sepal_width','petal_length','petal_width']]
y_train = train.species
X_test = test[['sepal_length','sepal_width','petal_length','petal_width']]
y_test = test.species

model = DecisionTreeClassifier(max_depth = 3, random_state = 1)
model.fit(X_train,y_train)

prediction=model.predict(X_test)

# Compute confusion matrix
labels = sorted(y_test.unique())
cm = metrics.confusion_matrix(y_test, prediction, labels=labels)

# Convert to DataFrame for pretty printing
cm_df = pd.DataFrame(cm, index=labels, columns=labels)

# Print report for CML
print(f"## Sanity Test Report")
print(f"**Model Accuracy:** {metrics.accuracy_score(prediction,y_test):.3f}%\n")
print("### Confusion Matrix (True vs Predicted):\n")
print(cm_df.to_markdown())

joblib.dump(model, "artifacts/model.joblib")
