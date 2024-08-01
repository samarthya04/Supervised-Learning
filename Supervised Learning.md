# Supervised Learning in Machine Learning

Supervised learning is a type of machine learning where the model is trained on a labeled dataset. The goal is to learn a mapping from inputs to outputs, which can then be used to make predictions on new, unseen data. Supervised learning can be categorized into two main types: classification and regression.

## Classification

In classification, the goal is to predict a discrete label or category for a given input. For example, determining whether an email is spam or not, or classifying a handwritten digit from an image.

### Example: Classifying Iris Species

Let's use the popular Iris dataset to demonstrate classification. The Iris dataset consists of measurements of iris flowers from three different species.

#### Step 1: Load the Data

```python
import pandas as pd
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
```

#### Step 2: Split the Data

```python
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X = df.drop(columns='species')
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

#### Logistic Regression
A linear model for binary classification that estimates the probability that an instance belongs to a particular class.
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Train Logistic Regression model
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = log_reg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Logistic Regression Accuracy: {accuracy:.2f}')
```

#### K-Nearest Neighbors
A non-parametric method that classifies instances based on the majority label of their nearest neighbors in the feature space.
```python
from sklearn.neighbors import KNeighborsClassifier

# Train K-Nearest Neighbors model
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'K-Nearest Neighbors Accuracy: {accuracy:.2f}')
```

#### Support Vector Machine
A powerful classifier that finds the hyperplane which maximizes the margin between different classes.
```python
from sklearn.svm import SVC

# Train Support Vector Machine model
svm = SVC()
svm.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Support Vector Machine Accuracy: {accuracy:.2f}')
```

#### Random Forest
An ensemble method that builds multiple decision trees and merges their results for more accurate and stable predictions.
```python
from sklearn.ensemble import RandomForestClassifier

# Train Random Forest model
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Random Forest Accuracy: {accuracy:.2f}')
```

## Regression

In regression, the goal is to predict a continuous value for a given input. For example, predicting the price of a house based on its features, or forecasting stock prices.

### Example: Predicting House Prices

Let's use a synthetic dataset to demonstrate regression. We'll predict house prices based on features like the number of rooms, lot size, and age of the house.

#### Step 1: Generate Synthetic Dataset

```python
import numpy as np
import pandas as pd

# Generate synthetic dataset
np.random.seed(42)
n_samples = 100
X = np.random.rand(n_samples, 3) * 100
y = X[:, 0] * 2000 + X[:, 1] * 5000 + X[:, 2] * 1000 + np.random.randn(n_samples) * 10000

# Convert to DataFrame
df = pd.DataFrame(X, columns=['rooms', 'lot_size', 'age'])
df['price'] = y
```

#### Step 2: Split the Data

```python
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X = df.drop(columns='price')
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

#### Linear Regression
A simple model that assumes a linear relationship between the input features and the target variable.
```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Train Linear Regression model
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = lin_reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Linear Regression Mean Squared Error: {mse:.2f}')
```

#### Decision Tree Regressor
A non-linear model that splits the data into regions and fits a constant value in each region.
```python
from sklearn.tree import DecisionTreeRegressor

# Train Decision Tree Regressor model
tree_reg = DecisionTreeRegressor()
tree_reg.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = tree_reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Decision Tree Regressor Mean Squared Error: {mse:.2f}')
```

#### Random Forest Regressor
An ensemble method that builds multiple decision trees and averages their predictions for better accuracy and stability.
```python
from sklearn.ensemble import RandomForestRegressor

# Train Random Forest Regressor model
rf_reg = RandomForestRegressor(n_estimators=100)
rf_reg.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = rf_reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Random Forest Regressor Mean Squared Error: {mse:.2f}')
```

#### Support Vector Regressor
A model that finds a hyperplane in the feature space which approximates the target values within a certain margin of tolerance.
```python
from sklearn.svm import SVR

# Train Support Vector Regressor model
svr = SVR()
svr.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = svr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Support Vector Regressor Mean Squared Error: {mse:.2f}')
```

## Conclusion

Supervised learning is a powerful technique for both classification and regression tasks. By training models on labeled data, we can make accurate predictions on new, unseen data. The choice of model and the quality of the data are crucial for achieving good performance. This guide provided basic examples of classification and regression using multiple algorithms with the Iris dataset and a synthetic dataset, respectively. Explore and experiment with different models and datasets to gain a deeper understanding of supervised learning.
