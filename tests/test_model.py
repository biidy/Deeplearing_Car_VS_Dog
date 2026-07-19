"""
Tests unitaires légers — volontairement rapides et sans GPU, pour pouvoir
tourner sur un runner GitHub Actions standard (CPU only).

On ne teste PAS ici la performance du modèle (accuracy, etc.), seulement
que le pipeline de code fonctionne correctement (formes de tenseurs,
absence d'erreurs de chargement). La validation de performance se fait
à part, sur Colab, avec MLrun pour le tracking.
"""

import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model import build_cnn_from_scratch, get_data_augmentation  # noqa: E402


def test_build_cnn_from_scratch_output_shape():
    """Le modèle doit produire une sortie de forme (batch, 1) — classification binaire."""
    model = build_cnn_from_scratch(input_shape=(150, 150, 3))
    dummy_input = np.random.rand(2, 150, 150, 3).astype("float32")
    output = model(dummy_input)
    assert output.shape == (2, 1)


def test_output_range_is_probability():
    """La sortie sigmoid doit toujours être comprise entre 0 et 1."""
    model = build_cnn_from_scratch(input_shape=(150, 150, 3))
    dummy_input = np.random.rand(1, 150, 150, 3).astype("float32")
    output = model(dummy_input).numpy()
    assert 0.0 <= output[0][0] <= 1.0


def test_data_augmentation_preserves_shape():
    """La data augmentation ne doit pas modifier la forme des images."""
    augmentation = get_data_augmentation()
    dummy_input = np.random.rand(1, 150, 150, 3).astype("float32")
    output = augmentation(dummy_input, training=True)
    assert output.shape == dummy_input.shape


def test_saved_model_loads_if_present():
    """
    Si un modèle entraîné est présent dans models/, vérifie qu'il se
    charge sans erreur et accepte une image factice.
    Ce test est ignoré si aucun modèle n'est encore committé.
    """
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "cat_dog_classifier.keras")
    if not os.path.exists(model_path):
        return  # pas d'échec si le modèle n'est pas encore présent

    model = tf.keras.models.load_model(model_path)
    dummy_input = np.random.rand(1, *model.input_shape[1:]).astype("float32")
    output = model.predict(dummy_input, verbose=0)
    assert output.shape == (1, 1)