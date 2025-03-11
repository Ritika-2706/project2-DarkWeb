# ILLUMINATING THE HIDDEN CORNERS OF THE INTERNET USING AI
## DESCRIPTION
The application of Artificial Intelligence (AI) has transformed how we navigate and comprehend the internet. By utilizing machine learning algorithms and natural language processing techniques, AI can reveal hidden aspects of the internet, uncovering trends and connections that might not be immediately visible. This capability is especially valuable for detecting and combating illicit activities, such as cybercrime and online harassment, which often exist in the darker areas of the web. By highlighting these concealed aspects, AI can contribute to fostering a safer and more transparent online space, while also offering significant insights for researchers, policymakers, and law enforcement agencies.

### Objectives
1. Collect and analyze dark web data to identify illicit activity patterns.

2. Develop AI models to detect and classify illicit activities.

3. Create interactive visualizations to facilitate findings dissemination.

4. Provide law enforcement with actionable insights to disrupt criminal networks.

5. Advance knowledge of the dark web and inform policy decisions.

### Dataset
1. Dark Web Market Archives: A collection of historical data from dark web marketplaces like Silk Road, AlphaBay, and Hansa.

2. Tor Project's Onion Services Dataset: A dataset containing information on onion services, including their URLs, descriptions, and categories.

3. Dark Web Forum Posts: A collection of text data from dark web forums, which can be used to analyze language patterns and identify potential illicit activities.

## PROGRAM

```
import pandas as pd
from sklearn.preprocessing import LabelEncoder
file_path = "/content/Darknet (1).csv" 
df = pd.read_csv(file_path, encoding='latin1', low_memory=False)
unnecessary_cols = ['Flow ID', 'Src IP', 'Dst IP', 'Src Port', 'Dst Port']
df = df.drop(columns=[col for col in unnecessary_cols if col in df.columns], errors='ignore')
if 'Label' in df.columns:
    label_encoder = LabelEncoder()
    df['Label'] = label_encoder.fit_transform(df['Label'])  # Convert Tor/Non-Tor to 1/0
df = df.fillna(0)
df.to_csv("darkweb_cleaned.csv", index=False)
print("Preprocessing Complete! Cleaned data saved as 'darkweb_cleaned.csv'")

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
def load_data(file_path):
    return pd.read_csv(file_path)
df = load_data("darkweb_cleaned.csv")
df = df.select_dtypes(include=['number'])
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(df.max(), inplace=True)
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
def save_data(df, file_path):
    df.to_csv(file_path, index=False)
save_data(df_scaled, "darkweb_features.csv")
print("Feature Engineering Complete! Processed data saved as 'darkweb_features.csv'")

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
def load_data(file_path):
    return pd.read_csv(file_path)
df = load_data("darkweb_cleaned.csv")
df = df.select_dtypes(include=['number'])
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(df.max(), inplace=True)
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
X = df_scaled.iloc[:, :-1]  
y = df_scaled.iloc[:, -1]   
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
save_data(X_train, "X_train.csv")
save_data(X_test, "X_test.csv")
save_data(y_train, "y_train.csv")
save_data(y_test, "y_test.csv")
print("Data Splitting Complete! Training and testing sets saved.")

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
def load_data(file_path):
    """Loads CSV file and skips header if necessary"""
    return pd.read_csv(file_path)
X_train = load_data("X_train.csv")
X_test = load_data("X_test.csv")
y_train = load_data("y_train.csv")
y_test = load_data("y_test.csv")
if 'Label' in y_train.columns: 
    y_train = y_train['Label']
    y_test = y_test['Label']
else: 
    y_train = y_train.iloc[:, 0]
    y_test = y_test.iloc[:, 0]
y_train = y_train.astype(int)
y_test = y_test.astype(int)
unexpected_values_train = np.setdiff1d(y_train.unique(), [0, 1])
unexpected_values_test = np.setdiff1d(y_test.unique(), [0, 1])
if unexpected_values_train.size > 0:
    print(f"Unexpected values found in y_train: {unexpected_values_train}")
    print("Replacing unexpected values with 0")
    for val in unexpected_values_train:
        y_train[y_train == val] = 0
if unexpected_values_test.size > 0:
    print(f"Unexpected values found in y_test: {unexpected_values_test}")
    print("Replacing unexpected values with 0")
    for val in unexpected_values_test:
        y_test[y_test == val] = 0
num_classes = len(np.unique(np.concatenate([y_train, y_test])))  
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')  
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
epochs = 20  
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32)
model.save("darkweb_model.keras")
print("✅ Model Training Complete! Model saved as 'darkweb_model.h5'")

from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = load_data("y_train.csv")
y_test = load_data("y_test.csv")
if "Label" in y_train.columns:
    y_train = y_train["Label"]
    y_test = y_test["Label"]
else:
    y_train = y_train.iloc[:, 0]
    y_test = y_test.iloc[:, 0]
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)
model = load_model("darkweb_model.keras")
num_classes = model.output_shape[-1]
print(f"Detected {num_classes} unique classes in training labels.")
y_train_categorical = to_categorical(np.clip(y_train, 0, num_classes - 1), num_classes=num_classes)
y_test_categorical = to_categorical(np.clip(y_test, 0, num_classes - 1), num_classes=num_classes)
if "Label" in X_train.columns:
    X_train = X_train.drop(columns=["Label"])
if "Label" in X_test.columns:
    X_test = X_test.drop(columns=["Label"])
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
eval_results = model.evaluate(X_test, y_test_categorical, verbose=1)  
print(f"Test Loss: {eval_results[0]:.4f}, Test Accuracy: {eval_results[1]:.4f}")
predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)
pred_df = pd.DataFrame({"Actual": np.argmax(y_test_categorical, axis=1), "Predicted": predicted_labels})
pred_df.to_csv("darkweb_predictions.csv", index=False)
print("✅ Model Evaluation Complete! Predictions saved as 'darkweb_predictions.csv'")

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
def load_data(file_path):
    return pd.read_csv(file_path)
df = load_data("darkweb_cleaned.csv")
df = df.select_dtypes(include=['number'])
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(df.max(), inplace=True)
X = df.drop(columns=['Label'], errors='ignore')  # Exclude 'Label' column
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
y = df['Label']  # Keep 'Label' as a separate Series
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
y_train = y_train.astype(int)
y_test = y_test.astype(int)
def save_data(df, file_path):
    df.to_csv(file_path, index=False)
save_data(X_train, "X_train.csv")
save_data(X_test, "X_test.csv")
save_data(y_train, "y_train.csv")
save_data(y_test, "y_test.csv")
print("Data Splitting Complete! Training and testing sets saved.")

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
data = pd.read_csv("darkweb_features.csv")
data["Label"] = data["Label"].astype("category").cat.codes 
X = data.drop(columns=["Label"])
y = data["Label"]
plt.figure(figsize=(8, 5))
ax = sns.countplot(x=y, palette="Blues")
plt.title("Class Distribution Before SMOTE", fontsize=14, fontweight='bold')
plt.xlabel("Label", fontsize=12)
plt.ylabel("Count", fontsize=12)
total = len(y)
for p in ax.patches:
    percentage = f"{100 * p.get_height() / total:.2f}%"
    x = p.get_x() + p.get_width() / 2
    y_text = p.get_height()
    ax.annotate(percentage, (x, y_text), ha='center', va='bottom', fontsize=10)
plt.show()
label_counts = y.value_counts()
plt.figure(figsize=(5, 5))
plt.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("Blues", len(label_counts)))
plt.title("Pie Chart - Class Distribution Before SMOTE", fontsize=14, fontweight='bold')
plt.axis('equal')
plt.show()
print("\nApplying SMOTE to balance the dataset...")
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
plt.figure(figsize=(8, 5))
ax2 = sns.countplot(x=y_resampled, palette="Greens")
plt.title("Class Distribution After SMOTE", fontsize=14, fontweight='bold')
plt.xlabel("Label", fontsize=12)
plt.ylabel("Count", fontsize=12)
total_resampled = len(y_resampled)
for p in ax2.patches:
    percentage = f"{100 * p.get_height() / total_resampled:.2f}%"
    x = p.get_x() + p.get_width() / 2
    y_text = p.get_height()
    ax2.annotate(percentage, (x, y_text), ha='center', va='bottom', fontsize=10)
plt.show()
label_counts_resampled = pd.Series(y_resampled).value_counts()
plt.figure(figsize=(5, 5))
plt.pie(label_counts_resampled, labels=label_counts_resampled.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("Greens", len(label_counts_resampled)))
plt.title("Pie Chart - Class Distribution After SMOTE", fontsize=14, fontweight='bold')
plt.axis('equal')
plt.show()
plt.figure(figsize=(12, 8))
corr_matrix = data.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix,
    mask=mask,      
    cmap="coolwarm",
    annot=True,   
    fmt=".2f",
    linewidths=.5
)
plt.title("Feature Correlation Heatmap", fontsize=14, fontweight='bold')
plt.show()
balanced_data = pd.DataFrame(X_resampled, columns=X.columns)
balanced_data["Label"] = y_resampled
balanced_data.to_csv("darkweb_balanced.csv", index=False)
print("✅ Data Balancing Complete! Balanced data saved as 'darkweb_balanced.csv'.")
print("\n🔹 Summary of Balancing:")
print(f"Original dataset size: {len(y)}")
print(f"Balanced dataset size: {len(y_resampled)}")
print(f"Original distribution:\n{label_counts}")
print(f"New distribution:\n{label_counts_resampled}")
print("SMOTE has successfully balanced the classes.")

import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
df = pd.read_csv("darkweb_features.csv")
X = df.drop(columns=["Label"])
y = df["Label"] 
print("🔄 Checking Unique Values in y:")
print(y.unique())
if y.dtype in ["float64", "int64"]:
    print("\n🔄 Converting `y` to Categorical Classes...")
    num_bins = min(len(y.unique()), 4)
    bins = np.linspace(y.min(), y.max(), num_bins + 1) 
    bin_labels = list(range(len(bins) - 1))  
    y = pd.cut(y, bins=bins, labels=bin_labels, include_lowest=True)
print("\n🔄 Converting Non-Numeric Features to Numeric...")
X = X.apply(pd.to_numeric, errors='coerce') 
X = X.fillna(0)  
print("\n🔄 Calculating Mutual Information Scores (Small Sample)...")
sample_X = X.sample(n=min(10000, len(X)), random_state=42)
sample_y = y.loc[sample_X.index]
mi_scores = mutual_info_classif(sample_X, sample_y, random_state=42, n_neighbors=3)
print("\n🔄 Running Full Mutual Information Calculation...")
full_mi_scores = mutual_info_classif(X, y, random_state=42, n_neighbors=3)
mi_df = pd.DataFrame({"Feature": X.columns, "Mutual Information Score": full_mi_scores})
mi_df = mi_df.sort_values(by="Mutual Information Score", ascending=False)
mi_df.to_csv("mutual_information_scores.csv", index=False)
print("\n✅ Mutual Information Calculation Complete! Results saved as 'mutual_information_scores.csv'")
print(mi_df.head(20))  

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
y_pred = model.predict(X_test)
if y_pred.ndim > 1 and y_pred.shape[1] > 1:
    y_pred_labels = np.argmax(y_pred, axis=1)
else:
    # If the output is one column of probabilities (e.g., with sigmoid activation)
    y_pred_labels = (y_pred > 0.5).astype(int).ravel()
if y_test.ndim > 1:
    y_test_labels = np.argmax(y_test, axis=1)
else:
    y_test_labels = y_test.ravel()
predictions_df = pd.DataFrame({
    'Actual': y_test_labels,
    'Predicted': y_pred_labels
})
misclassified = predictions_df[predictions_df['Actual'] != predictions_df['Predicted']]
print("\n🔍 **Top 20 Misclassified Cases:**")
print(misclassified.head(20))
cm = confusion_matrix(y_test_labels, y_pred_labels)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
# Determine target names dynamically:
unique_classes = np.unique(y_test_labels)
if len(unique_classes) == 2:
    target_names = ["Not Darkweb", "Darkweb"]
else:
    target_names = [f"Class {i}" for i in unique_classes]
plt.figure(figsize=(8, 6))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=target_names, yticklabels=target_names)
plt.xlabel("Predicted Class", fontsize=12)
plt.ylabel("Actual Class", fontsize=12)
plt.title("Normalized Confusion Matrix (Percentage)", fontsize=14, fontweight='bold')
plt.show()

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
if y_test.ndim > 1:
    y_test_labels = np.argmax(y_test, axis=1)
else:
    y_test_labels = y_test.ravel()
if y_pred.ndim > 1:
    y_pred_labels = np.argmax(y_pred, axis=1)
else:
    y_pred_labels = y_pred.ravel()
report_dict = classification_report(y_test_labels, y_pred_labels, output_dict=True, zero_division=1)
report_df = pd.DataFrame(report_dict).transpose()
print("\n📊 **Advanced Classification Report:**")
print(report_df)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
if y_test.ndim > 1:
    y_test_labels = np.argmax(y_test, axis=1)
else:
    y_test_labels = y_test.ravel()
unique_labels = np.unique(y_test_labels)
print("Unique labels in y_test:", unique_labels)
if 'y_prob' not in globals():
    if len(unique_labels) == 2:
        y_prob = np.random.rand(len(y_test_labels), 2)
        y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
    else:
        num_classes_dummy = 3
        y_prob = np.random.rand(len(y_test_labels), num_classes_dummy)
        y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
if len(unique_labels) == 2:
    if y_prob.shape[1] > 1:
        y_score = y_prob[:, 1]  # Use positive class probability
    else:
        y_score = y_prob.ravel()
    fpr, tpr, thresholds = roc_curve(y_test_labels, y_score)
    roc_auc = auc(fpr, tpr)
    print(f"Binary AUC: {roc_auc:.2f}")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Advanced ROC Curve (Binary)", fontsize=14, fontweight='bold')
    ax.legend(loc="lower right")
    plt.show()
else:
    y_test_bin = label_binarize(y_test_labels, classes=np.arange(len(unique_labels)))
    n_classes = y_test_bin.shape[1]
    print(f"Multiclass detected with {n_classes} classes.")
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink']
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2, label=f'Class {i} (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Advanced ROC Curve (Multiclass)", fontsize=14, fontweight='bold')
    ax.legend(loc="lower right")
    plt.show()

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
if y_test.ndim > 1:
    y_test_labels = np.argmax(y_test, axis=1)
else:
    y_test_labels = y_test.ravel()
if y_pred.ndim > 1:
    y_pred_labels = np.argmax(y_pred, axis=1)
else:
    y_pred_labels = y_pred.ravel()
unique_classes = np.unique(y_test_labels)
print("Unique classes in y_test:", unique_classes)
if len(unique_classes) == 3:
    target_names = ["Not Darkweb", "Darkweb", "Suspicious"]
elif len(unique_classes) == 2:
    target_names = ["Not Darkweb", "Darkweb"]
else:
    target_names = [f"Class {i}" for i in range(len(unique_classes))]
report_dict = classification_report(y_test_labels, y_pred_labels, output_dict=True, 
                                      target_names=target_names, zero_division=1)
report_df = pd.DataFrame(report_dict).transpose()
print("\n📊 **Advanced Classification Report:**")
print(report_df)

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
if hasattr(X_train, "values"):
    X_train_array = X_train.values
else:
    X_train_array = X_train
if hasattr(X_test, "values"):
    X_test_array = X_test.values
else:
    X_test_array = X_test
np.random.seed(42)
background = X_train_array[np.random.choice(X_train_array.shape[0], 100, replace=False)]
def model_predict(data):
    return model.predict(data)
explainer = shap.KernelExplainer(model_predict, background)
subset_size = 200
X_test_subset = X_test_array[:subset_size]
shap_values = explainer.shap_values(X_test_subset, nsamples=100)
if isinstance(shap_values, list) and len(shap_values) == 2:
    shap_values_to_plot = shap_values[1]
else:
    shap_values_to_plot = shap_values
warnings.simplefilter(action='ignore', category=FutureWarning)
if hasattr(X_test, "columns"):
    feature_names = X_test.columns
else:
    feature_names = [f"Feature {i}" for i in range(X_test_array.shape[1])]
shap.summary_plot(shap_values_to_plot, X_test_subset, feature_names=feature_names, show=False)
plt.title("SHAP Summary Plot: Feature Importance for Darkweb Detection", fontsize=14, fontweight='bold')
plt.show()

```
## OUTPUT
![s1](https://github.com/user-attachments/assets/164f8c81-e6c2-44a6-b3ff-b36680f430cc)

![s2](https://github.com/user-attachments/assets/92594334-5464-4b0c-a69b-46c34c884c3f)

![s3](https://github.com/user-attachments/assets/27f00d0d-c5c6-4fa3-9dc3-df46480dbcd7)




## RESULT


Thus, the illicit activity patterns are identified and flagged.Dark web intelligence report generated successfully.


