# Supervised Learning in Machine Learning

Supervised learning is a type of machine learning where the model is trained on a labeled dataset. The goal is to learn a mapping from inputs to outputs, which can then be used to make predictions on new, unseen data. Supervised learning can be categorized into two main types: classification and regression.

## Classification

In classification, the goal is to predict a discrete label or category for a given input. For example, determining whether an email is spam or not, or classifying a handwritten digit from an image.

### Algorithms Used

1. **Logistic Regression**: A linear model for binary classification that estimates the probability that an instance belongs to a particular class.
2. **K-Nearest Neighbors (KNN)**: A non-parametric method that classifies instances based on the majority label of their nearest neighbors in the feature space.
3. **Support Vector Machine (SVM)**: A powerful classifier that finds the hyperplane which maximizes the margin between different classes.
4. **Random Forest**: An ensemble method that builds multiple decision trees and merges their results for more accurate and stable predictions.

### Example: Classifying Iris Species

Let's use the popular Iris dataset to demonstrate classification. The Iris dataset consists of measurements of iris flowers from three different species.

### Code Snippet

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset
from sklearn.datasets import load_iris
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target

# Split the data into training and testing sets
X = df.drop(columns='species')
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=200),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Support Vector Machine': SVC(),
    'Random Forest': RandomForestClassifier(n_estimators=100)
}

# Train and evaluate each classifier
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{name} Accuracy: {accuracy:.2f}')
```

## Regression

In regression, the goal is to predict a continuous value for a given input. For example, predicting the price of a house based on its features, or forecasting stock prices.

### Algorithms Used

1. **Linear Regression**: A simple model that assumes a linear relationship between the input features and the target variable.
2. **Decision Tree Regressor**: A non-linear model that splits the data into regions and fits a constant value in each region.
3. **Random Forest Regressor**: An ensemble method that builds multiple decision trees and averages their predictions for better accuracy and stability.
4. **Support Vector Regressor (SVR)**: A model that finds a hyperplane in the feature space which approximates the target values within a certain margin of tolerance.

### Example: Predicting House Prices

Let's use a synthetic dataset to demonstrate regression. We'll predict house prices based on features like the number of rooms, lot size, and age of the house.

### Code Snippet

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

# Generate synthetic dataset
np.random.seed(42)
n_samples = 100
X = np.random.rand(n_samples, 3) * 100
y = X[:, 0] * 2000 + X[:, 1] * 5000 + X[:, 2] * 1000 + np.random.randn(n_samples) * 10000

# Convert to DataFrame
df = pd.DataFrame(X, columns=['rooms', 'lot_size', 'age'])
df['price'] = y

# Split the data into training and testing sets
X = df.drop(columns='price')
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize regressors
regressors = {
    'Linear Regression': LinearRegression(),
    'Decision Tree Regressor': DecisionTreeRegressor(),
    'Random Forest Regressor': RandomForestRegressor(n_estimators=100),
    'Support Vector Regressor': SVR()
}

# Train and evaluate each regressor
for name, reg in regressors.items():
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'{name} Mean Squared Error: {mse:.2f}')
```

## Conclusion

Supervised learning is a powerful technique for both classification and regression tasks. By training models on labeled data, we can make accurate predictions on new, unseen data. The choice of model and the quality of the data are crucial for achieving good performance. This guide provided basic examples of classification and regression using multiple algorithms with the Iris dataset and a synthetic dataset, respectively. Explore and experiment with different models and datasets to gain a deeper understanding of supervised learning.
