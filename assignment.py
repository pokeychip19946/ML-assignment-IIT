import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv('Fraud_check.csv')

# Create target variable
df['Risk'] = np.where(df['Taxable.Income'] <= 30000, 0, 1)

# Drop unnecessary columns
df.drop(columns=['City.Population', 'Taxable.Income'], inplace=True)

# Convert categorical variables to numerical
df = pd.get_dummies(df, drop_first=True)

# Split data into features and target
X = df.drop(columns=['Risk']).values  # Convert to numpy array
y = df['Risk'].values  # Convert to numpy array

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree Classifier
class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        if len(np.unique(y)) == 1 or (self.max_depth and depth >= self.max_depth) or n_samples < self.min_samples_split:
            return np.bincount(y).argmax()

        best_feature, best_threshold = self._best_split(X, y)
        if best_feature is None:
            return np.bincount(y).argmax()

        left_indices = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] > best_threshold

        left_subtree = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._grow_tree(X[right_indices], y[right_indices], depth + 1)

        return (best_feature, best_threshold, left_subtree, right_subtree)

    def _best_split(self, X, y):
        best_gini = float('inf')
        best_split = None

        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_indices = X[:, feature_index] <= threshold
                right_indices = X[:, feature_index] > threshold

                if len(y[left_indices]) < self.min_samples_leaf or len(y[right_indices]) < self.min_samples_leaf:
                    continue

                gini = self._gini_impurity(y[left_indices], y[right_indices])
                if gini < best_gini:
                    best_gini = gini
                    best_split = (feature_index, threshold)

        return best_split

    def _gini_impurity(self, left, right):
        total_samples = len(left) + len(right)
        if total_samples == 0:
            return 0

        p_left = len(left) / total_samples
        p_right = len(right) / total_samples

        gini_left = 1 - sum((np.bincount(left) / len(left)) ** 2) if len(left) > 0 else 0
        gini_right = 1 - sum((np.bincount(right) / len(right)) ** 2) if len(right) > 0 else 0

        return (p_left * gini_left) + (p_right * gini_right)

    def predict(self, X):
        return np.array([self._predict(sample, self.tree) for sample in X])

    def _predict(self, sample, tree):
        if not isinstance(tree, tuple):
            return tree

        feature_index, threshold, left_subtree, right_subtree = tree
        if sample[feature_index] <= threshold:
            return self._predict(sample, left_subtree)
        else:
            return self._predict(sample, right_subtree)

# Hyperparameter tuning with GridSearchCV
dt = DecisionTree()
param_grid = {
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create a decision tree classifier with sklearn for hyperparameter tuning
from sklearn.tree import DecisionTreeClassifier

# Prepare data for GridSearchCV
grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print(f'Best Hyperparameters: {best_params}')

# Fit the best model
best_tree = DecisionTree(max_depth=best_params['max_depth'],
                          min_samples_split=best_params['min_samples_split'],
                          min_samples_leaf=best_params['min_samples_leaf'])
best_tree.fit(X_train, y_train)

# Predictions and evaluation
y_pred = best_tree.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Classification Report:\n', report)

# Pruned Decision Tree Classifier
class PrunedDecisionTree(DecisionTree):
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        super().__init__(max_depth, min_samples_split, min_samples_leaf)

    def _prune_tree(self, tree, X, y):
        if not isinstance(tree, tuple):
            return tree

        feature_index, threshold, left_subtree, right_subtree = tree
        left_pred = self.predict(X[X[:, feature_index] <= threshold])
        right_pred = self.predict(X[X[:, feature_index] > threshold])
        
        left_count = np.bincount(left_pred)
        right_count = np.bincount(right_pred)
        
        if len(left_count) < self.min_samples_leaf or len(right_count) < self.min_samples_leaf:
            return np.bincount(y).argmax()

        return (feature_index, threshold, self._prune_tree(left_subtree, X[X[:, feature_index] <= threshold], y), 
                self._prune_tree(right_subtree, X[X[:, feature_index] > threshold], y))

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)
        self.tree = self._prune_tree(self.tree, X, y)

# Fit the pruned model
pruned_model = PrunedDecisionTree(max_depth=best_params['max_depth'],
                                   min_samples_split=best_params['min_samples_split'],
                                   min_samples_leaf=best_params['min_samples_leaf'])
pruned_model.fit(X_train, y_train)

# Make predictions with pruned model
y_pred_pruned = pruned_model.predict(X_test)

# Evaluate the pruned model
accuracy_pruned = accuracy_score(y_test, y_pred_pruned)
report_pruned = classification_report(y_test, y_pred_pruned)

print(f'Pruned Model Accuracy: {accuracy_pruned}')
print('Pruned Classification Report:\n', report_pruned)
