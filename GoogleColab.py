import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV, learning_curve
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
import seaborn as sns
import numpy as np
from math import sqrt

# Load dataset
df = pd.read_csv('DataFinal_Banget.csv')

# Menampilkan 5 baris pertama kerangka data
print(df.head())

# Periksa apakah ada nilai yang hilang dalam dataset
print(df.isnull().sum())

# Split dataset dengan fitur (X) dan variabel target (y)
X = df.drop(columns=['Diagnosa'])
y = df['Diagnosa']

# Split dataset
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=90)

# Inisialisasi variabel untuk menemukan random state & test size terbaik
best_random_state = None
best_split = None
best_score = np.inf

# simpan iterasi
iterations = []

# Looping random state dan test size
for random_state in range(1, 101):
    for test_size in [0.2, 0.3, 0.4]:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Train simple decision tree classifier
        clf = DecisionTreeClassifier(random_state=random_state)
        clf.fit(X_train, y_train)

        # Prediksi ke data testing
        y_pred = clf.predict(X_test)

        # Kalkulasi akurasi
        accuracy = accuracy_score(y_test, y_pred)

        # Hitung jumlah sampel dalam set training dan testing
        num_train_samples = X_train.shape[0]
        num_test_samples = X_test.shape[0]

        # Cek distribusi data
        train_dist = y_train.value_counts(normalize=True)
        test_dist = y_test.value_counts(normalize=True)

        # Hitung perbedaan distribusi antara set training dan testing
        distribution_difference = np.abs(train_dist - test_dist).sum()

        # Append hasil
        iterations.append({
            'random_state': random_state,
            'test_size': test_size,
            'distribution_difference': distribution_difference,
            'accuracy': accuracy,
            'num_train_samples': num_train_samples,
            'num_test_samples': num_test_samples
        })

        if distribution_difference < best_score:
            best_random_state = random_state
            best_split = test_size
            best_score = distribution_difference

# Sort dari yang terkecil
sorted_iterations = sorted(iterations, key=lambda x: x['distribution_difference'])

# Print semua iterasi
print("\nAll Iterations Sorted by Distribution Difference:")
for iteration in sorted_iterations:
    print(f"Random State: {iteration['random_state']}, Test Size: {iteration['test_size']}, Distribution Difference: {iteration['distribution_difference']:.6f}, Accuracy: {iteration['accuracy']:.4f}, Train Samples: {iteration['num_train_samples']}, Test Samples: {iteration['num_test_samples']}")

# 8 terbaik
top_8_iterations = sorted_iterations[:8]

# Print 8 terbaik
print("\nTop 8 Combinations of Test Size and Random State:")
for idx, iteration in enumerate(top_8_iterations):
    print(f"Combination {idx+1}: Random State: {iteration['random_state']}, Test Size: {iteration['test_size']}, Distribution Difference: {iteration['distribution_difference']:.6f}, Accuracy: {iteration['accuracy']:.4f}, Train Samples: {iteration['num_train_samples']}, Test Samples: {iteration['num_test_samples']}")

# Create a DataFrame
df_top_8 = pd.DataFrame(top_8_iterations)

# Plot the table
fig, ax = plt.subplots()
ax.axis('tight')
ax.axis('off')
ax.table(cellText=df_top_8.values, colLabels=df_top_8.columns, cellLoc='center', loc='center')
plt.show()

# Simpan iterasi ke excel
df_all_iterations = pd.DataFrame(sorted_iterations)
df_all_iterations.to_excel('Iterasi_Split_Data.xlsx', index=False)

# Gunakan keadaan random state dan test size untuk pemisahan akhir
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=best_split, random_state=best_random_state)

unique_labels = df['Diagnosa'].unique()
print("Unique labels in 'Diagnosa':", unique_labels)

X_train

y_train

# Cek distribusi data
print("Distribution of labels in the training set:")
print(y_train.value_counts(normalize=True))

print("\nDistribution of labels in the testing set:")
print(y_test.value_counts(normalize=True))

# Plot distribusi data
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
y_train.value_counts(normalize=True).plot(kind='bar', ax=axes[0], color='skyblue')
axes[0].set_title('Training Set Label Distribution')
axes[0].set_xlabel('Label')
axes[0].set_ylabel('Proportion')

y_test.value_counts(normalize=True).plot(kind='bar', ax=axes[1], color='salmon')
axes[1].set_title('Testing Set Label Distribution')
axes[1].set_xlabel('Label')

plt.tight_layout()
plt.show()

clf = DecisionTreeClassifier(random_state=best_random_state)
clf.fit(X_train, y_train)

importances = clf.feature_importances_
feature_names = X.columns

feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print(feature_importance_df)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importances')
plt.gca().invert_yaxis()
plt.show()

top_features = feature_importance_df[feature_importance_df['Importance'] > 0].Feature.tolist()
X_train_selected = X_train[top_features]
X_test_selected = X_test[top_features]

# Define parameter grid
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [int(x) for x in range(10, 110, 20)] + [None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None, 'auto', 'sqrt', 'log2']
}

# Inisasi GridSearchCV
grid_search = GridSearchCV(estimator=DecisionTreeClassifier(random_state=best_random_state),
                           param_grid=param_grid,
                           scoring='accuracy',
                           cv=5,
                           n_jobs=-1,
                           verbose=10)  # Set total iterasi

# Fit GridSearchCV ke data training
grid_search.fit(X_train_selected, y_train)

# Extract and tampilkan hyperparameter terbaik
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Display score terbaik
best_score = grid_search.best_score_
print("Best Score:", best_score)

# Print iterasi
cv_results = grid_search.cv_results_
for i in range(len(cv_results['params'])):
    print(f"Iteration {i + 1}")
    print(f"Params: {cv_results['params'][i]}")
    print(f"Mean Test Score: {cv_results['mean_test_score'][i]}")
    print(f"Rank: {cv_results['rank_test_score'][i]}")
    print('-' * 30)

# Ubah cv_results_ menjadi DataFrame
cv_results_df = pd.DataFrame(grid_search.cv_results_)

# Sort by rank_test_score dalam ascending
cv_results_df_sorted = cv_results_df.sort_values(by='rank_test_score', ascending=True)

# Remove duplicates pada 'mean_test_score'
cv_results_df_unique = cv_results_df_sorted.drop_duplicates(subset='mean_test_score', keep='first')

# Select top 10 unique results
top_10_results_unique = cv_results_df_unique.head(10)

# Tampilkantop 10 unique hyperparameters
print("Top 10 Unique Hyperparameters:")
with pd.option_context('display.max_colwidth', None):
    print(top_10_results_unique[['params', 'mean_test_score', 'rank_test_score']])

# Format tabel untuk hanya menampilkan parameter dan skornya
top_10_results_table_unique = top_10_results_unique[['params', 'mean_test_score', 'rank_test_score']]

# Display  tabel
print(top_10_results_table_unique)

# Simpan ke excel
cv_results_df.to_excel('Iterasi_grid_search.xlsx', index=False)
top_10_results_table_unique.to_excel('top_10_grid_search_results.xlsx', index=False)

# Filter hasil GridSearchCV untuk model C4.5 (menggunakan 'entropy' criterion)
c45_results = cv_results_df[cv_results_df['param_criterion'] == 'entropy']

# Sort by rank_test_score ascending
c45_results_sorted = c45_results.sort_values(by='rank_test_score', ascending=True)

# Remove duplicates based on 'mean_test_score'
c45_results_unique = c45_results_sorted.drop_duplicates(subset='mean_test_score', keep='first')

# Select top 10 unique results
top_10_c45_results = c45_results_unique.head(10)

# Tamplikan 10 unique hyperparameters untuk C4.5
print("Top 10 Unique Hyperparameters for C4.5:")

with pd.option_context('display.max_colwidth', None):
    print(top_10_c45_results[['params', 'mean_test_score', 'rank_test_score']])

# Filter hasil GridSearchCV untuk model C4.5 (menggunakan 'entropy' criterion)
c45_results = cv_results_df[cv_results_df['param_criterion'] == 'entropy']

# Sort by rank_test_score pada ascending
c45_results_sorted = c45_results.sort_values(by='rank_test_score', ascending=True)

# Ambil hyperparameter terbaik untuk C4.5
best_c45_params = c45_results_sorted.iloc[0]['params']

# Update criterion menjadi 'entropy'
best_c45_params['criterion'] = 'entropy'

# Create C4.5 classifier
# c45_dec = DecisionTreeClassifier(criterion='entropy', **{k: best_params[k] for k in best_params if k != 'criterion'}, random_state=best_random_state)
c45_dec = DecisionTreeClassifier(**best_c45_params, random_state=best_random_state)

# Fit model ke training data
c45_dec.fit(X_train_selected, y_train)

# Predict
y_predC45 = c45_dec.predict(X_test_selected)

# Evaluate model
print("C4.5 Model Evaluation:")
print("Confusion Matrix:")
cmc45 = confusion_matrix(y_test, y_predC45)
print(cmc45)

print("Accuracy:", accuracy_score(y_test, y_predC45))
print("Precision:", precision_score(y_test, y_predC45, average='weighted'))
print("Recall:", recall_score(y_test, y_predC45, average='weighted'))
print("F1 Score:", f1_score(y_test, y_predC45, average='weighted'))
print("Classification Report:")
print(classification_report(y_test, y_predC45))


# Plot confusion matrix
labels = ['Dermatitis', 'Gastritis', 'ISK', 'ISPA', 'Typhoid Fever']
disp_c45 = ConfusionMatrixDisplay(confusion_matrix=cmc45, display_labels=labels)
disp_c45.plot()
plt.title('Confusion Matrix for C4.5 Model')
plt.show()

# Stratified Cross Validation
skf = StratifiedKFold(n_splits=10)
cross_val_scores = cross_val_score(c45_dec, X_train_selected, y_train, cv=skf, scoring='accuracy')

print("Stratified Cross Validation Scores:", cross_val_scores)
print("Mean Cross Validation Score:", cross_val_scores.mean())
print("Standard Deviation of Cross Validation Score:", cross_val_scores.std())

# Calculate training accuracy
y_pred_trainC45 = c45_dec.predict(X_train_selected)
train_accuracy = accuracy_score(y_train, y_pred_trainC45)
print("Training Accuracy:", train_accuracy)

# Plot cross-validation scores
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cross_val_scores) + 1), cross_val_scores, marker='o', linestyle='--', color='b', label='Accuracy per fold')

# Plot mean cross-validation score
mean_cvscore = cross_val_scores.mean()
plt.axhline(y=mean_cvscore, color='r', linestyle='-', label=f'Mean CV Score: {mean_cvscore:.4f}')

# Plot test accuracy score
test_accuracy = accuracy_score(y_test, y_predC45)
plt.axhline(y=test_accuracy, color='g', linestyle='-', label=f'Test Accuracy: {test_accuracy:.4f}')

# Plot training accuracy score
plt.axhline(y=train_accuracy, color='orange', linestyle='-', label=f'Training Accuracy: {train_accuracy:.4f}')

plt.title('Cross-Validation Scores per Fold')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.xticks(range(1, len(cross_val_scores) + 1))
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Calculate errors
train_error = 1 - train_accuracy
test_error = 1 - accuracy_score(y_test, y_predC45)
cv_errors = 1 - cross_val_scores

# Plot cross-validation errors
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cv_errors) + 1), cv_errors, marker='o', linestyle='--', color='b', label='Error per fold')

# Plot mean cross-validation error
mean_cverror = 1 - cross_val_scores.mean()
plt.axhline(y=mean_cverror, color='r', linestyle='-', label=f'Mean CV Error: {mean_cverror:.4f}')

# Plot test error
plt.axhline(y=test_error, color='g', linestyle='-', label=f'Test Error: {test_error:.4f}')

# Plot training error
plt.axhline(y=train_error, color='orange', linestyle='-', label=f'Training Error: {train_error:.4f}')

plt.title('Cross-Validation Errors per Fold for C4.5 Model')
plt.xlabel('Fold')
plt.ylabel('Error')
plt.xticks(range(1, len(cv_errors) + 1))
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Filter hasil GridSearchCV untuk model CART (menggunakan 'gini' criterion)
cart_results = cv_results_df[cv_results_df['param_criterion'] == 'gini']

# Sort by rank_test_score dalam ascending
cart_results_sorted = cart_results.sort_values(by='rank_test_score', ascending=True)

# Remove duplicates based on 'mean_test_score'
cart_results_unique = cart_results_sorted.drop_duplicates(subset='mean_test_score', keep='first')

# Select top 10 unique results
top_10_cart_results = cart_results_unique.head(10)

# Display top 10 unique hyperparameters untuk CART
print("Top 10 Unique Hyperparameters for CART:")

with pd.option_context('display.max_colwidth', None):
    print(top_10_cart_results[['params', 'mean_test_score', 'rank_test_score']])

# Filter hasil GridSearchCV untuk model CART (menggunakan 'gini' criterion)
cart_results = cv_results_df[cv_results_df['param_criterion'] == 'gini']

# Sort by rank_test_score dalam ascenbding
cart_results_sorted = cart_results.sort_values(by='rank_test_score', ascending=True)

# Ambil hyperparameter terbaik untuk CART
best_cart_params = cart_results_sorted.iloc[0]['params']

# Update criterion menjadi 'gini'
best_cart_params['criterion'] = 'gini'

# Create CART classifier
# cart_dec = DecisionTreeClassifier(criterion='gini', **{k: best_params[k] for k in best_params if k != 'criterion'}, random_state=best_random_state)
cart_dec = DecisionTreeClassifier(**best_cart_params, random_state=best_random_state)

# Fit the model ke data training
cart_dec.fit(X_train_selected, y_train)

# Predict
y_pred_cart = cart_dec.predict(X_test_selected)

# Evaluate model
print("CART Model Evaluation:")
print("Confusion Matrix:")
cm_cart = confusion_matrix(y_test, y_pred_cart)
print(cm_cart)

print("Accuracy:", accuracy_score(y_test, y_pred_cart))
print("Precision:", precision_score(y_test, y_pred_cart, average='weighted'))
print("Recall:", recall_score(y_test, y_pred_cart, average='weighted'))
print("F1 Score:", f1_score(y_test, y_pred_cart, average='weighted'))
print("Classification Report:")
print(classification_report(y_test, y_pred_cart))

# Plot confusion matrix
labels = ['Dermatitis', 'Gastritis', 'ISK', 'ISPA', 'Typhoid Fever']
disp_cart = ConfusionMatrixDisplay(confusion_matrix=cm_cart, display_labels=labels)
disp_cart.plot()
plt.title('Confusion Matrix for CART Model')
plt.show()

# Stratified Cross Validation
skf = StratifiedKFold(n_splits=10)
cross_val_scores_cart = cross_val_score(cart_dec, X_train_selected, y_train, cv=skf, scoring='accuracy')

print("Stratified Cross Validation Scores:", cross_val_scores_cart)
print("Mean Cross Validation Score:", cross_val_scores_cart.mean())
print("Standard Deviation of Cross Validation Score:", cross_val_scores_cart.std())

# Calculate training accuracy
y_pred_train_cart = cart_dec.predict(X_train_selected)
train_accuracy_cart = accuracy_score(y_train, y_pred_train_cart)
print("Training Accuracy:", train_accuracy_cart)

# Plot cross-validation scores
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cross_val_scores_cart) + 1), cross_val_scores_cart, marker='o', linestyle='--', color='b', label='Accuracy per fold')

# Plot mean cross-validation score
mean_cv_score_cart = cross_val_scores_cart.mean()
plt.axhline(y=mean_cv_score_cart, color='r', linestyle='-', label=f'Mean CV Score: {mean_cv_score_cart:.4f}')

# Plot test accuracy score
test_accuracy_cart = accuracy_score(y_test, y_pred_cart)
plt.axhline(y=test_accuracy_cart, color='g', linestyle='-', label=f'Test Accuracy: {test_accuracy_cart:.4f}')

# Plot training accuracy score
plt.axhline(y=train_accuracy_cart, color='orange', linestyle='-', label=f'Training Accuracy: {train_accuracy_cart:.4f}')

plt.title('Cross-Validation Scores per Fold for CART Model')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.xticks(range(1, len(cross_val_scores_cart) + 1))
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Calculate errors
train_error_cart = 1 - train_accuracy_cart
test_error_cart = 1 - test_accuracy_cart
cv_errors_cart = 1 - cross_val_scores_cart

# Plot cross-validation errors
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cv_errors_cart) + 1), cv_errors_cart, marker='o', linestyle='--', color='b', label='Error per fold')

# Plot mean cross-validation error
mean_cv_error_cart = 1 - mean_cv_score_cart
plt.axhline(y=mean_cv_error_cart, color='r', linestyle='-', label=f'Mean CV Error: {mean_cv_error_cart:.4f}')

# Plot test error
plt.axhline(y=test_error_cart, color='g', linestyle='-', label=f'Test Error: {test_error_cart:.4f}')

# Plot training error
plt.axhline(y=train_error_cart, color='orange', linestyle='-', label=f'Training Error: {train_error_cart:.4f}')

plt.title('Cross-Validation Errors per Fold for CART Model')
plt.xlabel('Fold')
plt.ylabel('Error')
plt.xticks(range(1, len(cv_errors_cart) + 1))
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV, learning_curve
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
import seaborn as sns
import numpy as np
from math import sqrt
import joblib
from sklearn.tree import plot_tree


# Load dataset
df = pd.read_csv('DataFinal_Banget.csv')

# Menampilkan 5 baris pertama kerangka data
print(df.head())

# Periksa apakah ada nilai yang hilang dalam dataset
print(df.isnull().sum())

# Split dataset dengan fitur (X) dan variabel target (y)
X = df.drop(columns=['Diagnosa'])
y = df['Diagnosa']

# Split dataset
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=90)

# Initialize variables for finding the best random state and test size
best_random_state = None
best_split = None
best_score = np.inf

# List to store results for all iterations
iterations = []

# Loop through random states and test sizes
for random_state in range(1, 101):
    for test_size in [0.2, 0.3, 0.4]:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Train a simple decision tree classifier
        clf = DecisionTreeClassifier(random_state=random_state)
        clf.fit(X_train, y_train)

        # Predict on test set
        y_pred = clf.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)

        # Calculate number of samples in training and testing sets
        num_train_samples = X_train.shape[0]
        num_test_samples = X_test.shape[0]

        # Check the distribution of labels in the training and testing sets
        train_dist = y_train.value_counts(normalize=True)
        test_dist = y_test.value_counts(normalize=True)

        # Calculate the difference in distribution between training and testing sets
        distribution_difference = np.abs(train_dist - test_dist).sum()

        # Store the current iteration results
        iterations.append({
            'random_state': random_state,
            'test_size': test_size,
            'distribution_difference': distribution_difference,
            'accuracy': accuracy,
            'num_train_samples': num_train_samples,
            'num_test_samples': num_test_samples
        })

        # We want the distribution difference to be as small as possible
        if distribution_difference < best_score:
            best_random_state = random_state
            best_split = test_size
            best_score = distribution_difference

# Sort all iterations by the smallest distribution difference
sorted_iterations = sorted(iterations, key=lambda x: x['distribution_difference'])

# Print all iterations sorted by distribution difference
print("\nAll Iterations Sorted by Distribution Difference:")
for iteration in sorted_iterations:
    print(f"Random State: {iteration['random_state']}, Test Size: {iteration['test_size']}, Distribution Difference: {iteration['distribution_difference']:.6f}, Accuracy: {iteration['accuracy']:.4f}, Train Samples: {iteration['num_train_samples']}, Test Samples: {iteration['num_test_samples']}")

# Select the top 8 combinations
top_8_iterations = sorted_iterations[:8]

# Print the top 8 combinations
print("\nTop 8 Combinations of Test Size and Random State:")
for idx, iteration in enumerate(top_8_iterations):
    print(f"Combination {idx+1}: Random State: {iteration['random_state']}, Test Size: {iteration['test_size']}, Distribution Difference: {iteration['distribution_difference']:.6f}, Accuracy: {iteration['accuracy']:.4f}, Train Samples: {iteration['num_train_samples']}, Test Samples: {iteration['num_test_samples']}")

# Create a DataFrame of the top 8 combinations for plotting purposes
df_top_8 = pd.DataFrame(top_8_iterations)

# Plot the table
fig, ax = plt.subplots()
ax.axis('tight')
ax.axis('off')
ax.table(cellText=df_top_8.values, colLabels=df_top_8.columns, cellLoc='center', loc='center')
plt.show()

# Use the best random state and test size for final split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=best_split, random_state=best_random_state)

# Display unique labels in the 'Diagnosa' column
unique_labels = df['Diagnosa'].unique()
print("Unique labels in 'Diagnosa':", unique_labels)

X_train

y_train

# Check the distribution of labels in the training and testing sets
print("Distribution of labels in the training set:")
print(y_train.value_counts(normalize=True))

print("\nDistribution of labels in the testing set:")
print(y_test.value_counts(normalize=True))

# Plot the distribution of labels
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
y_train.value_counts(normalize=True).plot(kind='bar', ax=axes[0], color='skyblue')
axes[0].set_title('Training Set Label Distribution')
axes[0].set_xlabel('Label')
axes[0].set_ylabel('Proportion')

y_test.value_counts(normalize=True).plot(kind='bar', ax=axes[1], color='salmon')
axes[1].set_title('Testing Set Label Distribution')
axes[1].set_xlabel('Label')

plt.tight_layout()
plt.show()

clf = DecisionTreeClassifier(random_state=best_random_state)
clf.fit(X_train, y_train)

importances = clf.feature_importances_
feature_names = X.columns

feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print(feature_importance_df)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importances')
plt.gca().invert_yaxis()
plt.show()

top_features = feature_importance_df[feature_importance_df['Importance'] > 0].Feature.tolist()
X_train_selected = X_train[top_features]
X_test_selected = X_test[top_features]

# Define parameter grid
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [int(x) for x in range(10, 110, 20)] + [None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None, 'auto', 'sqrt', 'log2']
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=DecisionTreeClassifier(random_state=best_random_state),
                           param_grid=param_grid,
                           scoring='accuracy',
                           cv=5,
                           n_jobs=-1,
                           verbose=10)  # Set verbose to show progress

# Fit GridSearchCV to the training data
grid_search.fit(X_train_selected, y_train)

# Extract and display the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Display the best score
best_score = grid_search.best_score_
print("Best Score:", best_score)

# Print each iteration results
cv_results = grid_search.cv_results_
for i in range(len(cv_results['params'])):
    print(f"Iteration {i + 1}")
    print(f"Params: {cv_results['params'][i]}")
    print(f"Mean Test Score: {cv_results['mean_test_score'][i]}")
    print(f"Rank: {cv_results['rank_test_score'][i]}")
    print('-' * 30)

# Convert cv_results_ to DataFrame
cv_results_df = pd.DataFrame(grid_search.cv_results_)

# Sort by rank_test_score in ascending order
cv_results_df_sorted = cv_results_df.sort_values(by='rank_test_score', ascending=True)

# Remove duplicates based on 'mean_test_score' while keeping the first occurrence
cv_results_df_unique = cv_results_df_sorted.drop_duplicates(subset='mean_test_score', keep='first')

# Select top 10 unique results
top_10_results_unique = cv_results_df_unique.head(10)

# Display the top 10 unique hyperparameters
print("Top 10 Unique Hyperparameters:")
with pd.option_context('display.max_colwidth', None):
    print(top_10_results_unique[['params', 'mean_test_score', 'rank_test_score']])

# Format the table to show only the parameters and their scores
top_10_results_table_unique = top_10_results_unique[['params', 'mean_test_score', 'rank_test_score']]

# Display the table
print(top_10_results_table_unique)

# Filter hasil GridSearchCV untuk model CART (menggunakan 'gini' criterion)
cart_results = cv_results_df[cv_results_df['param_criterion'] == 'gini']

# Sort by rank_test_score in ascending order
cart_results_sorted = cart_results.sort_values(by='rank_test_score', ascending=True)

# Remove duplicates based on 'mean_test_score' while keeping the first occurrence
cart_results_unique = cart_results_sorted.drop_duplicates(subset='mean_test_score', keep='first')

# Select top 10 unique results
top_10_cart_results = cart_results_unique.head(10)

# Display the top 10 unique hyperparameters for CART
print("Top 10 Unique Hyperparameters for CART:")

with pd.option_context('display.max_colwidth', None):
    print(top_10_cart_results[['params', 'mean_test_score', 'rank_test_score']])

# Filter hasil GridSearchCV untuk model CART (menggunakan 'gini' criterion)
cart_results = cv_results_df[cv_results_df['param_criterion'] == 'gini']

# Sort by rank_test_score in ascending order
cart_results_sorted = cart_results.sort_values(by='rank_test_score', ascending=True)

# Ambil hyperparameter terbaik untuk CART
best_cart_params = cart_results_sorted.iloc[0]['params']

# Update criterion menjadi 'gini'
best_cart_params['criterion'] = 'gini'

# Create CART classifier
# cart_dec = DecisionTreeClassifier(criterion='gini', **{k: best_params[k] for k in best_params if k != 'criterion'}, random_state=best_random_state)
cart_dec = DecisionTreeClassifier(**best_cart_params, random_state=best_random_state)

# Fit the model on the training data
cart_dec.fit(X_train_selected, y_train)

# Predict on the test data
y_pred_cart = cart_dec.predict(X_test_selected)

# Evaluate the model
print("CART Model Evaluation:")
print("Confusion Matrix:")
cm_cart = confusion_matrix(y_test, y_pred_cart)
print(cm_cart)

print("Accuracy:", accuracy_score(y_test, y_pred_cart))
print("Precision:", precision_score(y_test, y_pred_cart, average='weighted'))
print("Recall:", recall_score(y_test, y_pred_cart, average='weighted'))
print("F1 Score:", f1_score(y_test, y_pred_cart, average='weighted'))
print("Classification Report:")
print(classification_report(y_test, y_pred_cart))

# Plot confusion matrix
labels = ['Dermatitis', 'Gastritis', 'ISK', 'ISPA', 'Typhoid Fever']
disp_cart = ConfusionMatrixDisplay(confusion_matrix=cm_cart, display_labels=labels)
disp_cart.plot()
plt.title('Confusion Matrix for CART Model')
plt.show()

# Stratified Cross Validation
skf = StratifiedKFold(n_splits=10)
cross_val_scores_cart = cross_val_score(cart_dec, X_train_selected, y_train, cv=skf, scoring='accuracy')

print("Stratified Cross Validation Scores:", cross_val_scores_cart)
print("Mean Cross Validation Score:", cross_val_scores_cart.mean())
print("Standard Deviation of Cross Validation Score:", cross_val_scores_cart.std())

# Calculate training accuracy
y_pred_train_cart = cart_dec.predict(X_train_selected)
train_accuracy_cart = accuracy_score(y_train, y_pred_train_cart)
print("Training Accuracy:", train_accuracy_cart)

# Plot the cross-validation scores
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cross_val_scores_cart) + 1), cross_val_scores_cart, marker='o', linestyle='--', color='b', label='Accuracy per fold')

# Plot mean cross-validation score
mean_cv_score_cart = cross_val_scores_cart.mean()
plt.axhline(y=mean_cv_score_cart, color='r', linestyle='-', label=f'Mean CV Score: {mean_cv_score_cart:.4f}')

# Plot test accuracy score
test_accuracy_cart = accuracy_score(y_test, y_pred_cart)
plt.axhline(y=test_accuracy_cart, color='g', linestyle='-', label=f'Test Accuracy: {test_accuracy_cart:.4f}')

# Plot training accuracy score
plt.axhline(y=train_accuracy_cart, color='orange', linestyle='-', label=f'Training Accuracy: {train_accuracy_cart:.4f}')

plt.title('Cross-Validation Scores per Fold for CART Model')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.xticks(range(1, len(cross_val_scores_cart) + 1))
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Calculate errors
train_error_cart = 1 - train_accuracy_cart
test_error_cart = 1 - test_accuracy_cart
cv_errors_cart = 1 - cross_val_scores_cart

# Plot the cross-validation errors
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cv_errors_cart) + 1), cv_errors_cart, marker='o', linestyle='--', color='b', label='Error per fold')

# Plot mean cross-validation error
mean_cv_error_cart = 1 - mean_cv_score_cart
plt.axhline(y=mean_cv_error_cart, color='r', linestyle='-', label=f'Mean CV Error: {mean_cv_error_cart:.4f}')

# Plot test error
plt.axhline(y=test_error_cart, color='g', linestyle='-', label=f'Test Error: {test_error_cart:.4f}')

# Plot training error
plt.axhline(y=train_error_cart, color='orange', linestyle='-', label=f'Training Error: {train_error_cart:.4f}')

plt.title('Cross-Validation Errors per Fold for CART Model')
plt.xlabel('Fold')
plt.ylabel('Error')
plt.xticks(range(1, len(cv_errors_cart) + 1))
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Path untuk menyimpan model
model_path = 'cart_model.joblib'

# Simpan model ke file
joblib.dump(cart_dec, model_path)
print(f"Model saved to {model_path}")

