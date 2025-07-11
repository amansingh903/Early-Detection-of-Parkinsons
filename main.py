import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

data = pd.read_excel("All data (M+F).xlsx")
target_map = {
    'Healthy': 0, 'H': 0,
    'Parkinsons': 1, "Parkinson's": 1, 'PD': 1, 'P': 1, 'Parkinson': 1
}
data['target'] = data['H+PD'].map(target_map)
data = data.dropna(subset=['target'])

features = [col for col in data.columns if col not in ['H+PD', 'target']]
X = data[features]
y = data['target'].astype(int)

X = X.dropna()
y = y.loc[X.index]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

classifiers = {
    "MLP": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42),
    "SVM": SVC(C=0.1, gamma='scale', kernel='rbf'),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "LogReg": LogisticRegression(max_iter=1000, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

for label, clf in classifiers.items():
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print(f"\n{label}")
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Healthy', 'Parkinson'])
    disp.plot(cmap='Blues')
    plt.title(f"Confusion Matrix: {label}")
    plt.show()
