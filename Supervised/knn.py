# K-Nearest neighbours algorithm

# Idea: Finds the "k" closest data points (neighbours) to given input
# and predicts based on majority class

# Distance measure: Euclidean distance 
# d = √ [ (x₁ - y₁)² + (x₂ - y₂)² + ... + (xₙ - yₙ)² 

import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KNN():

    def __init__(self, k = 3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self, x):
        # Compute distances
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # Get K nearest samples/labels
        k_indicies = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indicies]

        # Majority vote/most common class label
        most_common = Counter(k_nearest_labels).most_common(1)

        return most_common[0][0]



# Example test
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)

model = KNN(k=5)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy:.2f}")


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Reduce dimensionality to 2D for visualization
pca = PCA(n_components=2)
X_train_2d = pca.fit_transform(X_train)
X_test_2d = pca.transform(X_test)

# Create color map for classes
colors = ['red', 'green', 'blue']
target_names = data.target_names

# Plot training data
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
for class_idx in np.unique(y_train):
    plt.scatter(
        X_train_2d[y_train == class_idx, 0],
        X_train_2d[y_train == class_idx, 1],
        label=target_names[class_idx],
        color=colors[class_idx],
        alpha=0.6
    )
    
plt.title("Training Data (PCA 2D)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()

# Plot test predictions
plt.subplot(1, 2, 2)
for class_idx in np.unique(predictions):
    plt.scatter(
        X_test_2d[predictions == class_idx, 0],
        X_test_2d[predictions == class_idx, 1],
        label=f'Predicted {target_names[class_idx]}',
        color=colors[class_idx],
        edgecolor='k',
        marker='o',
        alpha=0.8
    )

plt.title("Test Predictions (PCA 2D)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()

plt.tight_layout()
plt.show()
