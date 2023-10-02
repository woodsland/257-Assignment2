# COMP257 - Unsupervised & Reinforcement Learning (Section 002)
# Assignment 2 - k-Means and DBSCAN
# Name: Wai Lim Leung
# ID  : 301276989
# Date: 30-Sep-2023

from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_samples
from matplotlib.ticker import FixedLocator, FixedFormatter
import warnings
warnings.filterwarnings("ignore")


# Part 1 - Retrieve & Load Olivetti Faces
olivetti_faces = fetch_olivetti_faces(shuffle=True, random_state=42)
of_data = olivetti_faces.data
of_target = olivetti_faces.target

print()
print("Part 1 - Olivetti Faces")
print("Data Shape  :", of_data.shape)
print("Target Shape:", of_target.shape)

fig = plt.figure(figsize=(6, 2))
for i in range(3):
    face_image = of_data[i].reshape(64, 64)
    position = fig.add_subplot(1, 3, i + 1)
    position.imshow(face_image, cmap='gray')
    position.set_title(f"Person {of_target[i]}")
    position.axis('off')

plt.tight_layout()
plt.show()


# Part 2a - Training 60%
X_train, X_temp, y_train, y_temp = train_test_split(of_data, of_target, test_size=0.4, random_state=42, stratify=of_target)

# Part 2b - Validation 20% & Test each 20%
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Part 2c - Output
print()
print("Part 2 - Training, Validation and Test Set Size")
print("Training  :", X_train.shape[0])
print("Validation:", X_val.shape[0])
print("Test set  :", X_test.shape[0])


# Part 3 - kFold & Predict
print()
print("Part 3 - kFold & Predict")

svm_classifier = SVC(kernel='linear', C=1)
n_folds = 5
stratified_kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
cross_val_scores = cross_val_score(svm_classifier, X_train, y_train, cv=stratified_kfold, scoring='accuracy')

for fold, accuracy in enumerate(cross_val_scores, start=1):
    print("Accuracy - Fold",  fold, ":", accuracy)

mean_accuracy = cross_val_scores.mean()
print()
print("Accuracy - Mean              : ", mean_accuracy)

std_accuracy = cross_val_scores.std()
print("Accuracy - Standard Deviation: ", std_accuracy)

svm_classifier.fit(X_train, y_train)
validation_accuracy = svm_classifier.score(X_val, y_val)
print("Accuracy - Validation Set    : ", validation_accuracy)


# Part 4 - k-Means
print()
print("Part 4 - k-Means")
cluster_range = range(2, 11)
silhouette_scores = []

# Perform K-Means clustering for different numbers of clusters
for n_clusters in cluster_range:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(of_data)
    print("Inertia for ncluster", n_clusters, ":", kmeans.inertia_)

    # Calculate the silhouette score for this clustering
    silhouette_avg = silhouette_score(of_data, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Silhouette Scores / Numbers of Clusters
plt.figure(figsize=(8, 4))
plt.plot(cluster_range, silhouette_scores, marker='o', linestyle='-')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score / Number of Clusters')

# Set the grid lines below the data points
plt.gca().set_axisbelow(True)
plt.grid()

# Best clusters with highest silhouette score
best_n_clusters = cluster_range[np.argmax(silhouette_scores)]
print()
print("Best Number of Clusters: ", best_n_clusters)

# K-Means clustering with best number of clusters
kmeans = KMeans(n_clusters=best_n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(of_data)

# Plot cluster centers
plt.figure(figsize=(8, 6))
plt.scatter(of_data[:, 0], of_data[:, 1], c=cluster_labels, s=1, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='red', marker='x')
plt.xlabel("$x_1$")
plt.ylabel("$x_2$", rotation=0)
plt.title(f"K-means Clustering (Best Cluster k={best_n_clusters})")
plt.gca().set_axisbelow(True)
plt.grid()

# Plot Elbow
kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(of_data)
                for k in range(2, 11)]
inertias = [model.inertia_ for model in kmeans_per_k]

plt.figure(figsize=(8, 6))
plt.plot(range(2, 11), inertias, "bo-")
plt.xlabel("$k$")
plt.ylabel("Inertia")
plt.annotate("", xy=(2, 26654), xytext=(3, 26600),
             arrowprops=dict(facecolor='black', shrink=0.1))
plt.text(3.4, 26610, "Elbow", horizontalalignment="center")
plt.grid()
plt.title("inertia_vs_k_plot")

# Silhouette Analysis Plot
plt.figure(figsize=(11, 9))

for k in (2, 3, 4, 5):
    plt.subplot(2, 2, k - 1)

    y_pred = kmeans_per_k[k - 1].labels_
    silhouette_coefficients = silhouette_samples(of_data, y_pred)

    padding = len(of_data) // 30
    pos = padding
    ticks = []
    for i in range(k):
        coeffs = silhouette_coefficients[y_pred == i]
        coeffs.sort()

        color = plt.cm.Spectral(i / k)
        plt.fill_betweenx(np.arange(pos, pos + len(coeffs)), 0, coeffs,
                          facecolor=color, edgecolor=color, alpha=0.7)
        ticks.append(pos + len(coeffs) // 2)
        pos += len(coeffs) + padding

    plt.gca().yaxis.set_major_locator(FixedLocator(ticks))
    plt.gca().yaxis.set_major_formatter(FixedFormatter(range(k)))
    if k in (2, 4):
        plt.ylabel("Cluster")

    if k in (4, 5):
        plt.gca().set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        plt.xlabel("Silhouette Coefficient")
    else:
        plt.tick_params(labelbottom=False)

    plt.axvline(x=silhouette_scores[k - 2], color="red", linestyle="--")
    plt.title(f"$k={k}$")

plt.show()


# Part 5 - Use best clusters to train Part 3
print()
print("Part 5 - Use best clusters to train Part 3")

# Assign cluster labels. Convert it from Part 4 to one-hot encoded format
encoder = OneHotEncoder(sparse=False)
cluster_labels_one_hot = encoder.fit_transform(cluster_labels.reshape(-1, 1))
augmented_data = np.hstack((of_data, cluster_labels_one_hot))

# Split into training, validation, and test sets
X_train_part5, X_temp_part5, y_train, y_temp = train_test_split(augmented_data, of_target, test_size=0.4, random_state=42, stratify=of_target)
X_val_part5, X_test_part5, y_val, y_test = train_test_split(X_temp_part5, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Train SVM classifier with training set
svm_classifier_part5 = SVC(kernel='linear', C=1)
svm_classifier_part5.fit(X_train_part5, y_train)

# Output accuracy
validation_accuracy_part5 = svm_classifier_part5.score(X_val_part5, y_val)
test_accuracy_part5 = svm_classifier_part5.score(X_test_part5, y_test)
print("Accuracy - Validation: ", validation_accuracy_part5)
print("Accuracy - Test      : ", test_accuracy_part5)


# Part 6 DBSCAN
print()
print("Part 6 - DBSCAN Clustering")

# X, y = of_data(n_samples=240, noise=0.05, random_state=42)
X, y = of_data, of_target
dbscan = DBSCAN(eps=0.05, min_samples=5)
dbscan.fit(X)

# Print the cluster labels
print("Labels                       : ", dbscan.labels_[:10])
print("Indices of the core instances: ", dbscan.core_sample_indices_[:10])

# Plot the results
def plot_dbscan(dbscan, X, size, show_xlabels=True, show_ylabels=True):
    core_mask = np.zeros_like(dbscan.labels_, dtype=bool)
    core_mask[dbscan.core_sample_indices_] = True
    anomalies_mask = dbscan.labels_ == -1
    non_core_mask = ~(core_mask | anomalies_mask)
    cores = dbscan.components_
    anomalies = X[anomalies_mask]
    non_cores = X[non_core_mask]

    plt.scatter(cores[:, 0], cores[:, 1], c=dbscan.labels_[core_mask], marker='o', s=size, cmap="Paired")
    plt.scatter(cores[:, 0], cores[:, 1], marker='*', s=20, c=dbscan.labels_[core_mask])
    plt.scatter(anomalies[:, 0], anomalies[:, 1], c="r", marker="x", s=100)
    plt.scatter(non_cores[:, 0], non_cores[:, 1], c=dbscan.labels_[non_core_mask], marker=".")
    if show_xlabels:
        plt.xlabel("$x_1$")
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("$x_2$", rotation=0)
    else:
        plt.tick_params(labelleft=False)

    plt.title(f"eps={dbscan.eps:.2f}, min_samples={dbscan.min_samples}")
    plt.grid()
    plt.gca().set_axisbelow(True)

# Set eps=10
dbscan2 = DBSCAN(eps=10.5)
dbscan2.fit(X)

fig = plt.figure(figsize=(9, 3.2))
plt.subplot(121)
plot_dbscan(dbscan, X, size=60)

plt.subplot(122)
plot_dbscan(dbscan2, X, size=240, show_ylabels=False)
plt.show()
