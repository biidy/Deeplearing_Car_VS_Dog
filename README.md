# 🐱🐶 Cat vs Dog Classifier — Deep Learning

Projet de classification d'images (chat vs chien) avec CNN et Transfer Learning,
déployé sous forme d'interface web interactive.

## Contexte

Ce projet implémente et compare plusieurs approches de deep learning pour
la classification d'images :
- Un CNN entraîné from scratch
- Du transfer learning avec des modèles préentraînés sur ImageNet
  (VGG16, ResNet50, InceptionV3)
- Du fine-tuning sur ces modèles préentraînés

Dataset : [Dogs vs Cats — Kaggle](https://www.kaggle.com/c/dogs-vs-cats/data)

## Structure du projet

```
cat-dog-classifier/
├── README.md
├── requirements.txt
├── .gitignore
├── .gitattributes               # config Git LFS pour le modèle
├── .github/
│   └── workflows/
│       └── deploy-to-hf.yml     # CI/CD : tests + déploiement vers HF Spaces
├── src/
│   ├── data_preprocessing.py    # organisation, nettoyage et chargement du dataset
│   ├── model.py                  # définitions des architectures (CNN, transfer learning)
│   └── train.py                  # script d'entraînement (exécuté à part, sur Colab/GPU)
├── models/
│   └── cat_dog_classifier.keras # modèle entraîné, versionné via Git LFS
├── app/
│   └── app.py                    # interface Gradio pour prédictions en temps réel
└── tests/
    └── test_model.py             # tests légers exécutés par le CI (CPU only)
```

**Important** : l'entraînement (`src/train.py`) ne tourne jamais dans GitHub Actions — les runners standards n'ont pas de GPU. L'entraînement se fait à part (Colab), et seul le modèle déjà entraîné est committé dans `models/` puis déployé automatiquement par le CI.

## Installation

```bash
git clone https://github.com/TON_USERNAME/cat-dog-classifier.git
cd cat-dog-classifier
pip install -r requirements.txt
```

## Utilisation

### 1. Préparer les données

```python
from src.data_preprocessing import organiser_dataset, nettoyer_images_corrompues

organiser_dataset('data/train_raw', 'data')
nettoyer_images_corrompues('data/train')
nettoyer_images_corrompues('data/validation')
```

### 2. Entraîner un modèle

```bash
python src/train.py --model resnet50 --epochs 15 --fine-tune
```

Modèles disponibles : `scratch`, `vgg16`, `resnet50`, `inception_v3`

### 3. Lancer l'interface locale

```bash
python app/app.py
```

## Déploiement (CI/CD automatique)

Le modèle entraîné sur Colab est téléchargé puis placé dans `models/`, et versionné
via **Git LFS** (nécessaire car il dépasse souvent la limite de 100 Mo de GitHub) :

```bash
git lfs install
git lfs track "models/*.keras"
git add .gitattributes models/cat_dog_classifier.keras
git commit -m "Ajout du modèle entraîné"
git push
```

À chaque push sur `main` touchant `app/`, `models/` ou `requirements.txt`, le workflow
`.github/workflows/deploy-to-hf.yml` :
1. exécute les tests unitaires (`tests/test_model.py`) sur un runner CPU standard
2. si les tests passent, pousse automatiquement `app/`, `models/` et `requirements.txt`
   vers le Space Hugging Face

Ça nécessite un secret GitHub nommé `HF_TOKEN` (token d'accès Hugging Face avec droits
d'écriture), à ajouter dans **Settings → Secrets and variables → Actions** du repo.

L'interface est ensuite accessible sur :
👉 `https://huggingface.co/spaces/TON_USERNAME/cat-dog-classifier`

## Résultats

| Modèle | Accuracy (validation) |
|---|---|
| CNN from scratch | à compléter |
| VGG16 (transfer learning) | à compléter |
| ResNet50 (transfer learning) | à compléter |
| InceptionV3 (transfer learning) | à compléter |

## Technologies

Python · TensorFlow / Keras · OpenCV · Gradio · Hugging Face Spaces · MLrun · Docker · Kubernetes

## Compétences démontrées

- Implémentation de CNN
- Transfer Learning & Fine-Tuning
- Optimisation des hyperparamètres (Grid/Random Search)
- Déploiement de modèle sous forme d'API / interface web
- Tracking d'expériences avec MLrun