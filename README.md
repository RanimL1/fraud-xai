# Détection de fraude explicable — IEEE-CIS (Challenge 2)

Application **Streamlit** de détection de fraude **explicable** : un modèle LightGBM entraîné sur le jeu IEEE-CIS (Vesta), des explications SHAP locales et globales, et un mode « et si ? » pour tester l'effet d'une modification sur le score.

## Contenu

```
fraud-xai/
├── notebooks/fraud_xai.ipynb   # projet de bout en bout : compréhension, qualité, préparation (avec graphiques), modèle, évaluation, SHAP, export
├── src/pipeline.py             # chargement, variables, découpage temporel, entraînement, export
├── train.py                    # même pipeline en ligne de commande
├── streamlit_app.py            # application Streamlit (point d'entrée)
├── src/engine.py               # inférence + explications TreeSHAP exactes (utilisé par Streamlit)
├── app/                        # variante optionnelle : API FastAPI + page HTML
├── data/                       # y déposer les CSV Kaggle (non versionnés)
└── artifacts/                  # modèle, métadonnées, figures (générés)
```

## 1. Données

Télécharger sur Kaggle (*IEEE-CIS Fraud Detection*, accepter le règlement de la compétition) puis placer dans `data/` :
`train_transaction.csv` et `train_identity.csv` (les fichiers `test_*` ne servent pas : ils n'ont pas d'étiquettes).

## 2. Entraînement

**Option A — Google Colab (recommandé, plus de mémoire)** : copier le dossier dans `MyDrive/fraud-xai`, ouvrir `notebooks/fraud_xai.ipynb`, tout exécuter. Télécharger ensuite le dossier `artifacts/` généré et le replacer dans le projet local.

**Option B — en local (PowerShell)** :
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-train.txt
python train.py              # ou : python train.py --nrows 100000 pour tester vite
```
Compter environ 8 Go de RAM pour le jeu complet.

## 3. Lancer l'application Streamlit

```powershell
pip install -r requirements.txt
streamlit run streamlit_app.py
```
Deux onglets : **Vue d'ensemble** (métriques de test, importance globale SHAP, matrice de confusion) et **Transactions et explications** (filtres, sélection d'une ligne, graphique SHAP local, scénarios « et si ? »).

## 4. Déployer sur Streamlit Community Cloud

1. Entraîner avec un échantillon léger : `python train.py --sample-size 5000` (ou exécuter le notebook, puis réduire `n_sample`).
2. Vérifier que `artifacts/model.txt` et `artifacts/sample.parquet` pèsent chacun moins de 100 Mo, puis les pousser sur GitHub (ils ne sont pas ignorés).
3. Sur share.streamlit.io : *New app* → choisir le dépôt, branche `main`, fichier principal `streamlit_app.py`. Le `requirements.txt` du dépôt suffit (il ne contient que le nécessaire à l'exécution).

Les données Kaggle ne doivent **pas** être publiées : seuls le modèle et l'échantillon de démonstration le sont. Vérifier que la licence de la compétition autorise cette redistribution d'un extrait.

## Variante : API FastAPI

```powershell
uvicorn app.main:app --reload
```
Interface HTML sur http://127.0.0.1:8000, documentation de l'API sur `/docs`. Routes : `/api/summary`, `/api/transactions`, `/api/explain/{idx}`, `/api/whatif`.

## Choix méthodologiques

- **Découpage temporel** (72 % / 8 % / 20 %) : le test est strictement postérieur à l'entraînement, comme en production.
- **Pas de fuite** : les encodages sont appris sur le train seul ; les variables d'historique par carte n'utilisent que le passé.
- **Déséquilibre (~3,5 % de fraudes)** : évaluation par PR-AUC, précision/rappel et un **coût métier** (fraude manquée = montant perdu, fausse alerte = 5 $). Seuils choisis sur la validation.
- **Explicabilité** : valeurs de Shapley exactes pour arbres (`pred_contrib` de LightGBM = TreeSHAP), en log-odds ; graphiques globaux (beeswarm, barres, dépendance) et locaux (waterfall) dans le notebook.

## Limites

Variables `V*`, `C*`, `D*` anonymisées (explication statistique, non causale) ; dérive de la fraude dans le temps ; coût d'une fausse alerte supposé ; échantillon de démonstration de l'application enrichi en fraudes (~20 %), alors que les métriques sont calculées sur tout le jeu de test.

## Publier sur GitHub

```powershell
git init; git add .; git commit -m "Détection de fraude explicable IEEE-CIS"
git branch -M main; git remote add origin <url-du-dépôt>; git push -u origin main
```
Les CSV Kaggle sont ignorés par `.gitignore`.
