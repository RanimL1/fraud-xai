# Détection de fraude explicable — IEEE-CIS 

Application **Streamlit multi-pages** : chaque phase du projet est une page, des données brutes à l'explication d'une décision individuelle. Modèle LightGBM entraîné sur le jeu IEEE-CIS (Vesta), explications SHAP, scénarios « et si ? ».

## Pages de l'application

| Page | Contenu |
|---|---|
| Accueil | résumé, indicateurs clés, déroulé du projet, limites |
| 1 · Données | sources, jointure transactions + identité, types, mémoire |
| 2 · Qualité | valeurs manquantes (par famille, informatives ?), anomalies, doublons, montants |
| 3 · Exploration | déséquilibre, temps, montant, produit, heure, appareil, corrélations |
| 4 · Préparation | variables dérivées, découpage temporel, encodage, récapitulatif des décisions |
| 5 · Modélisation | LightGBM, hyperparamètres, courbe d'apprentissage, importance « gain » |
| 6 · Évaluation | ROC, PR, distribution des scores, **seuil et coût interactifs**, stabilité par semaine, **comparaison avec le non supervisé** (Isolation Forest, ACP + K-Means) |
| 7 · Explicabilité globale | importance SHAP, nuage SHAP, dépendance par variable |
| 8 · Explicabilité locale | cas typiques ou liste de transactions, graphique SHAP, scénarios « et si ? » |

## Contenu du dépôt

```
fraud-xai/
├── streamlit_app.py            # point d'entrée (navigation entre les pages)
├── app_pages/                  # une page par phase + common.py (graphiques partagés)
├── src/pipeline.py             # chargement, variables, découpage, entraînement, export
├── src/profiling.py            # statistiques agrégées exportées pour les pages (aucune donnée brute)
├── src/engine.py               # inférence + explications TreeSHAP exactes
├── src/unsupervised.py         # Isolation Forest, ACP, K-Means (sans étiquettes) + comparaison au supervisé
├── train.py                    # pipeline complet en ligne de commande
├── notebooks/fraud_xai.ipynb   # le projet de bout en bout, avec graphiques à chaque étape
├── app/                        # variante optionnelle : API FastAPI + page HTML
├── data/                       # y déposer les CSV Kaggle (non versionnés)
└── artifacts/                  # modèle, profil, échantillon, figures (générés)
```

## 1. Données

Télécharger sur Kaggle (*IEEE-CIS Fraud Detection*, accepter le règlement de la compétition) et placer dans `data/` :
`train_transaction.csv` et `train_identity.csv`.

## 2. Entraînement

**Option A — Google Colab (recommandé, plus de mémoire)** : copier le dossier dans `MyDrive/fraud-xai`, ouvrir `notebooks/fraud_xai.ipynb`, tout exécuter. Télécharger ensuite `artifacts/` et le replacer dans le projet local.

**Option B — en local (PowerShell)** :
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-train.txt
python train.py              # ou : python train.py --nrows 100000 pour tester vite
```

Les deux voies produisent les mêmes fichiers dans `artifacts/` : `model.txt`, `meta.json`, `sample.parquet`, `profile.json`, `shap_sample.parquet`, `unsupervised.json` (et les figures pour le notebook).

## 3. Lancer l'application

```powershell
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## 4. Déployer sur Streamlit Community Cloud

1. Entraîner avec un échantillon léger : `python train.py --sample-size 5000`.
2. Vérifier que chaque fichier de `artifacts/` pèse moins de 100 Mo, puis les pousser sur GitHub (ils ne sont pas ignorés).
3. Sur share.streamlit.io : *New app* → dépôt, branche `main`, fichier principal `streamlit_app.py`.

Les pages de données s'appuient sur `profile.json` (agrégats uniquement) : les CSV Kaggle ne sont jamais nécessaires pour l'application et ne doivent **pas** être publiés. Vérifier que la licence de la compétition autorise la redistribution de l'échantillon de démonstration.

## Choix méthodologiques

- **Découpage temporel** (72 / 8 / 20 %) : le test est strictement postérieur à l'entraînement.
- **Pas de fuite** : encodages appris sur le train seul ; variables d'historique par carte calculées à partir du passé uniquement.
- **Déséquilibre (~3,5 % de fraudes)** : PR-AUC, précision/rappel et coût métier (fraude manquée = montant perdu, fausse alerte = 5 $, modifiable page 6).
- **Valeurs manquantes** conservées : le manquant est informatif et LightGBM le gère.
- **Explicabilité** : valeurs de Shapley exactes pour arbres (`pred_contrib` de LightGBM = TreeSHAP), en log-odds.

## Partie non supervisée

Deux méthodes ajustées **sans étiquettes** (`isFraud` ne sert qu'à l'évaluation) : **Isolation Forest** sur les variables numériques (manquants < 50 %, imputation par la médiane, catégories encodées exclues) et **ACP + K-Means** (8 segments). On compare leur capacité à faire remonter la fraude (ROC-AUC, lift sur les 5 % les plus anormaux) à LightGBM, puis on mesure la complémentarité : part des fraudes manquées par le modèle supervisé qui sont parmi les plus anormales. Résultat attendu : AUC bien inférieure au supervisé, mais un signal utile en filet de sécurité.

Le fichier `artifacts/unsupervised.json` fourni a été calculé sur l'échantillon de démonstration ; `python train.py` le régénère sur le jeu complet (ajusté sur le train, évalué sur le test).

## Limites

Variables `V*`, `C*`, `D*` anonymisées (explication statistique, non causale) ; dérive de la fraude dans le temps ; coût d'une fausse alerte supposé ; échantillon de démonstration enrichi en fraudes (~20 %) alors que les métriques portent sur tout le jeu de test.

## Auteur
 
Ranim Latrous, étudiante en 3e année, spécialité intelligence artificielle, ENSI.

