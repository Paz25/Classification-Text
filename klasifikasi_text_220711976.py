# %% [markdown]
# # Part 1: Single-Label Classification

# %%
import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# %%
train_data = pd.read_csv('./data/train_preprocess.csv')
val_data = pd.read_csv('./data/valid_preprocess.csv')
test_data = pd.read_csv('./data/test_preprocess.csv')

print('Train Data:', train_data.shape)
print('Validation Data:', val_data.shape)
print('Test Data:', test_data.shape)

# %%
train_data.head()

# %%
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text

train_data['clean_text'] = train_data['sentence'].apply(clean_text)
val_data['clean_text'] = val_data['sentence'].apply(clean_text)
test_data['clean_text'] = test_data['sentence'].apply(clean_text)

# %%
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, KFold

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_data['clean_text'])
X_val = vectorizer.transform(val_data['clean_text'])
X_test = vectorizer.transform(test_data['clean_text'])

# Target Labels
y_train = train_data['fuel']
y_val = val_data['fuel']
y_test = test_data['fuel']

param_grids = {
    'SVM': {
        'model': SVC(),
        'params': {
            'kernel': ['linear', 'rbf'],
            'C': [0.1, 1, 10]
        }
    },
    'KNN': {
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors':  [3, 5, 7],
            'weights': ['uniform', 'distance']
        }
    },
    'NB': {
        'model': MultinomialNB(),
        'params': {
            'alpha': [0.1, 0.5, 1.0]
        }
    }
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)

best_models = {}
results = {}

for model_name, config in param_grids.items():
    print(f"Running GridSearchCV for {model_name}...")

    grid  = GridSearchCV(config['model'], config['params'], cv=kf, scoring='accuracy')

    grid.fit(X_train, y_train)

    best_models[model_name] = grid.best_estimator_
    results[model_name] = grid.best_score_

    print(f"Best Params for {model_name}: {grid.best_params_}")
    print(f"Best Cross-Validation Accuracy: {grid.best_score_:.4f}\n")

# %%
# plot every model accuracy
label = ['negative', 'neutral', 'positive']
plt.figure(figsize=(10, 6))
plt.bar(results.keys(), results.values())
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.show()

for model_name, model in best_models.items():
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(f"{model_name} Accuracy on Test Set: {accuracy_score(y_test, y_pred):.4f}")
    print(f"{model_name} Classification Report:")
    print(classification_report(y_test, y_pred))

    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True,
                fmt='d', cmap='Blues', xticklabels=label, yticklabels=label)
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# %%
# Predict the test set
best_model = best_models['SVM']
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
test_data['predicted'] = y_pred
test_data

# %%
correct = test_data[test_data['fuel'] == test_data['predicted']]
incorrect = test_data[test_data['fuel'] != test_data['predicted']]
print(f"Correct Predictions: {len(correct)}")
print(f"Incorrect Predictions: {len(incorrect)}")

# %% [markdown]
# # Part 2: Multi-Label Classification

# %%
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.metrics import multilabel_confusion_matrix

# %%
labels = ['fuel', 'machine', 'others', 'part', 'price', 'service']

train_encoded = pd.get_dummies(train_data, columns=labels, dtype=int)
val_encoded = pd.get_dummies(val_data, columns=labels, dtype=int)
test_encoded = pd.get_dummies(test_data, columns=labels, dtype=int)

# %%
train_encoded.head()

# %%
label_columns = ['fuel_negative', 'fuel_neutral', 'fuel_positive', 'machine_negative', 'machine_neutral', 'machine_positive']

y_train_multi = train_encoded[label_columns]
y_val_multi = val_encoded[label_columns]
y_test_multi = test_encoded[label_columns]

# %%
from sklearn.model_selection import KFold, cross_val_score
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.multiclass import OneVsOneClassifier
import numpy as np

multi_kf = KFold(n_splits=5, shuffle=True, random_state=42)

results = {}

models = {"SVM": SVC(), "KNN": KNeighborsClassifier(), "NB": MultinomialNB()}

for model_name, model in models.items():
    wrapped_model = BinaryRelevance(model)

    scores = cross_val_score(
        wrapped_model, 
        X_train, 
        y_train_multi, 
        cv=multi_kf, 
        scoring='accuracy',
        n_jobs=1,
    )

    results[model_name] = scores
    print(
        f"{model_name} - Cross-Validation Accuracy: {np.mean(scores):.4f} +/- {np.std(scores):.4f}"
    )

    wrapped_model.fit(X_train, y_train_multi)

    y_pred_multi = wrapped_model.predict(X_test)

    accuracy = accuracy_score(y_test_multi, y_pred_multi)
    print(f"{model_name} Final Accuracy:", accuracy)

    print(f"{model_name} Classification Report")
    print(
        classification_report(
            y_test_multi, 
            y_pred_multi, 
            target_names=label_columns, 
            zero_division=1
        )
    )

    print(f"{model_name} Confusion Matrix:")
    mcm = multilabel_confusion_matrix(y_test_multi, y_pred_multi)
    rows, cols = 2, 3
    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 4, rows * 4)
    )
    axes = axes.flatten()

    for i, (ax, label) in enumerate(zip(axes, label_columns)):
        sns.heatmap(mcm[i], annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title(f"Confusion Matrix for {label}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    for j in range(len(label_columns), rows * cols):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

# %%
from scipy.sparse import csr_matrix
y_pred_test = {}
for model_name, model in models.items():
    wrapped_model = BinaryRelevance(model)
    wrapped_model.fit(X_train, y_train_multi)
    y_pred_test[model_name] = wrapped_model.predict(X_test)

for model_name, y_pred in y_pred_test.items():
    print(f"\n{model_name} Predictions:\n")
    dense_array = y_pred.toarray()
    print(dense_array)


