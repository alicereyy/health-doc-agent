# 🧠 RAG Assistant – Troubles du sommeil

Ce projet est un **prototype de Retrieval-Augmented Generation (RAG)** appliqué à des documents médicaux sur les troubles du sommeil.

Il permet de poser des questions en langage naturel sur un corpus de PDF, en s’appuyant sur :
- une base vectorielle (ChromaDB)
- des embeddings Sentence-Transformers
- un LLM OpenAI via LangChain

## 🚀 Fonctionnalités

- 📄 Ingestion de documents PDF
- ✂️ Découpage en chunks avec chevauchement
- 🔢 Embeddings vectoriels
- 🔍 Recherche par similarité avec score
- 🤖 Génération de réponses à partir du contexte
- 📚 Affichage des sources et scores
- 🌍 Choix de la langue de réponse (français / anglais)
- 📤 Upload de nouveaux PDF via une interface Streamlit

## 🛠️ Installation

### Créer un environnement virtuel avec les dépendances

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Configuration de la clé OpenAI

Le projet utilise l’API OpenAI via LangChain.
La clé doit être accessible via les variables d’environnement.

#### Option 1 — Clé déjà définie sur le système

Si la variable OPENAI_API_KEY est déjà définie, aucune action supplémentaire n’est nécessaire.

#### Option 2 — Fichier .env

Créer un fichier .env à la racine du projet :
```txt
OPENAI_API_KEY=sk-...
```
Le projet utilise python-dotenv pour charger automatiquement cette clé au démarrage.

### Construction de la base vectorielle

Avant de lancer l’application, il faut indexer les documents PDF présents dans le dossier data/ :
```bash
python build_index.py
```

Ce script :
- charge les PDF
- découpe les documents en chunks
- génère les embeddings
- crée une base ChromaDB persistante

## Lancer l’application
```bash
streamlit run app/streamlit_app.py
```
Depuis l’interface :
- poser des questions sur les documents
- consulter les sources utilisées
- uploader de nouveaux PDF (indexation incrémentale)

## ⚠️ Avertissement

Ce projet est un prototype pédagogique.
Il ne constitue pas un outil médical et ne remplace pas un avis professionnel.