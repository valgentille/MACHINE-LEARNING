# MACHINE-LEARNING
# 🤖 TP Machine Learning — Classification Algorithms

Trois travaux pratiques d'introduction au Machine Learning supervisé avec **scikit-learn**.

---

## 📁 Structure du projet

```
TPMachine-Learning/
├── tp1_decision_tree.ipynb       # Arbre de décision (Iris)
├── tp2_k_nearest_neighbours.ipynb # K plus proches voisins (Breast Cancer)
├── tp3_naive_bayes.ipynb         # Naive Bayes (Car Evaluation)
└── car_evaluation.csv            # Dataset pour le TP3
```

---

## 📚 Contenu des TPs

### TP1 — Decision Tree (Arbre de décision)
**Dataset :** Iris (sklearn built-in) — 150 échantillons, 3 classes de fleurs  
**Algorithme :** `DecisionTreeClassifier`

- Exploration des données (EDA) : statistiques, heatmap de corrélation, pairplot
- Entraînement avec critère **Gini** et critère **Entropy**
- Comparaison train/test accuracy, détection de l'overfitting
- Visualisation de l'arbre avec `sklearn.tree` et `graphviz`
- Matrice de confusion

---

### TP2 — K-Nearest Neighbours (KNN)
**Dataset :** Breast Cancer (sklearn built-in) — 569 échantillons, classification binaire  
**Algorithme :** `KNeighborsClassifier`

- EDA : rapport de corrélation personnalisé, visualisation des features
- Sélection du meilleur `k` (1 → 15) via courbe train/test accuracy
- Rapport de classification (précision, rappel, F1-score)
- Matrice de confusion
- Validation croisée (cross-validation, 5 folds)
- Courbe ROC et AUC

---

### TP3 — Naive Bayes
**Dataset :** Car Evaluation (`car_evaluation.csv`) — 1728 échantillons, 4 classes  
**Algorithme :** `GaussianNB`

- Gestion des valeurs manquantes
- Encodage des variables catégorielles : **Ordinal Encoding** puis **One-Hot Encoding**
- Comparaison des performances selon la méthode d'encodage
- Matrice de confusion, rapport de classification
- Validation croisée (10 folds)

---

## 🛠️ Installation

```bash
pip install numpy pandas matplotlib seaborn scikit-learn graphviz
```

> **Note :** Le TP3 utilise le fichier `car_evaluation.csv` — assurez-vous qu'il est dans le même dossier que le notebook.

---

## 🚀 Lancement

```bash
jupyter notebook
```

Ou avec Google Colab : importer les fichiers `.ipynb` et le CSV dans l'espace de fichiers.

---

## 📊 Résultats obtenus

| TP | Algorithme | Dataset | Accuracy (test) |
|----|-----------|---------|----------------|
| TP1 | Decision Tree (Gini) | Iris | ~96% |
| TP2 | KNN (k=8) | Breast Cancer | ~94% |
| TP3 | Naive Bayes (One-Hot) | Car Evaluation | ~85% |

---

## 📖 Références

- [scikit-learn Documentation](https://scikit-learn.org/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Seaborn Documentation](https://seaborn.pydata.org/)
