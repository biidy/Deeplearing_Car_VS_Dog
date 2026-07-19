"""
Préparation du dataset Dogs vs Cats :
- réorganisation en dossiers train/validation par classe
- nettoyage des images corrompues
- chargement en tf.data.Dataset prêt pour l'entraînement
"""

import os
import random
import shutil

import tensorflow as tf


def organiser_dataset(source_dir: str, dest_dir: str, split_ratio: float = 0.8) -> None:
    """
    Réorganise les images plates (cat.0.jpg, dog.0.jpg, ...) en dossiers
    train/validation par classe, format attendu par image_dataset_from_directory.

    Le split train/validation est fait AVANT tout traitement pour éviter
    toute fuite de données entre les deux ensembles.
    """
    random.seed(42)  # reproductibilité du split

    for classe in ["cat", "dog"]:
        fichiers = [f for f in os.listdir(source_dir) if f.startswith(classe)]
        random.shuffle(fichiers)

        n_train = int(len(fichiers) * split_ratio)

        for i, f in enumerate(fichiers):
            subset = "train" if i < n_train else "validation"
            dossier_cible = os.path.join(dest_dir, subset, classe + "s")
            os.makedirs(dossier_cible, exist_ok=True)
            shutil.copyfile(
                os.path.join(source_dir, f),
                os.path.join(dossier_cible, f),
            )

    print(f"Dataset organisé dans '{dest_dir}' (split {split_ratio:.0%}/{1 - split_ratio:.0%}).")


def nettoyer_images_corrompues(dossier_racine: str) -> None:
    """
    Parcourt récursivement toutes les images et supprime celles que
    TensorFlow ne peut pas décoder correctement. Nécessaire car le dataset
    Dogs vs Cats contient quelques fichiers .jpg invalides qui font
    échouer l'entraînement en plein milieu d'une epoch.
    """
    n_supprimees = 0
    n_total = 0

    for sous_dossier, _, fichiers in os.walk(dossier_racine):
        for fichier in fichiers:
            chemin = os.path.join(sous_dossier, fichier)
            n_total += 1
            try:
                img_bytes = tf.io.read_file(chemin)
                img = tf.io.decode_image(img_bytes, channels=3)
                if img.shape[0] == 0 or img.shape[1] == 0:
                    raise ValueError("image vide")
            except Exception as e:
                print(f"Corrompu -> suppression : {chemin} ({e})")
                os.remove(chemin)
                n_supprimees += 1

    print(f"Terminé : {n_supprimees}/{n_total} images supprimées dans '{dossier_racine}'.")


def charger_datasets(
    data_dir: str = "data",
    img_size: tuple = (150, 150),
    batch_size: int = 32,
):
    """
    Charge les datasets train et validation depuis la structure de dossiers
    data/train/{cats,dogs} et data/validation/{cats,dogs}.
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "train"),
        image_size=img_size,
        batch_size=batch_size,
        label_mode="binary",
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "validation"),
        image_size=img_size,
        batch_size=batch_size,
        label_mode="binary",
    )

    # Améliore les performances de chargement (chevauchement I/O et calcul)
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=autotune)
    val_ds = val_ds.prefetch(buffer_size=autotune)

    return train_ds, val_ds


if __name__ == "__main__":
    organiser_dataset("data/train_raw", "data")
    nettoyer_images_corrompues("data/train")
    nettoyer_images_corrompues("data/validation")