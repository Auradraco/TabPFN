#  Copyright (c) Prior Labs GmbH 2025.
"""Example of using TabPFN for binary classification.

This example demonstrates how to use TabPFNClassifier on a binary classification task
using the breast cancer dataset from scikit-learn.
"""
from jinja2 import Template
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, classification_report, log_loss
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNClassifier
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load data
X, y = load_breast_cancer(return_X_y=True)

# Things to note for breast cancer dataset:
# Data contains 30 columns (features) and 569 rows (samples)
# Data can be found from https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
# So the classification of Diagnosis: "malignant" and "benign" is test against the 30 features 
# M : class 0, B: class 1


# print("Values of x: ", X)
# print("Length of x: ", len(X))
# print("Length of features for one sample: ", len(X[0]))
# # print("Values of y: ", y)
# # print(type(X[0]))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Initialize a classifier
clf = TabPFNClassifier()
clf.fit(X_train, y_train)

# Predict probabilities
prediction_probabilities = clf.predict_proba(X_test)
print("ROC AUC:", roc_auc_score(y_test, prediction_probabilities[:, 1]))
print("Log Loss: ", log_loss(y_test, prediction_probabilities))



# Predict labels
predictions = clf.predict(X_test)
print("Accuracy", accuracy_score(y_test, predictions))

# Predict F1 score
# print("Test samples: ", X_test)
# print("Test Labels: ", y_test)
predictions = clf.predict(X_test)
# print("Predictions: ", predictions)
print("F1 Score", f1_score(y_test, predictions))



# confusion = confusion_matrix(y_test, predictions, labels=clf.classes_)
# plt.figure(figsize=(10, 8))

# sns.heatmap(confusion, 
#             annot=True, 
#             fmt='d', 
#             cmap= 'flare',
#             xticklabels=["Malignant", "Benign"],
#             yticklabels=["Malignant", "Benign"],
#             annot_kws={"size": 35, "color": "black"}
#             )
# plt.xlabel('Predicted label', fontsize=20)
# plt.ylabel('True label', fontsize=20)
# plt.title('Confusion Matrix of Breast Cancer Classification', fontsize=24)
# # plt.show()


# savedir = "results/"
# confusion_filename = "confusion_matrix_breast_cancer.png"
# confusion_path = os.path.join(savedir, confusion_filename)
# plt.savefig(confusion_path)
# plt.close()
# # print(f"Confusion Matrix: \n{confusion}")

# comparison = "Malignant vs Benign"

# class_report = classification_report(y_test, predictions, target_names=["Malignant", "Benign"], digits=4, output_dict=True)
# class_report_df = pd.DataFrame(class_report).transpose()

# report_html = class_report_df.to_html()

# html_sections = []
# html_sections.append(f"""
#                 <h3>Confusion Matrix</h2>
#                 <img src="{confusion_filename}" width="600">
#                 <h3> Accuracy Table </h3>
#                 {report_html}
#                      """)

# html_template = Template("""
#         <!DOCTYPE html>
#         <html lang="en">
#             <head>
#                 <title> TabPFN Binary Breast Cancer Classification Report </title>
#             </head>
#             <body>
#                 <h1>TabPFN Binary Breast Cancer Classification Report</h1>
#                 <h2>Performance Matrix</h1>
#                 {% for section in sections %}
#                 {{section}}
#                 {% endfor %}

#             </body>
                         
#                          """)

# html_output = html_template.render(
#     sections = html_sections,
# )

# html_filename = savedir + "tabpfn_breast_cancer_classification_report.html"
# with open(html_filename, "w") as f:
#     f.write(html_output)
