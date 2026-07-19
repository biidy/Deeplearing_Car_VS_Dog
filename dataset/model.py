"""
Définitions des architectures :
- CNN from scratch
- Transfer learning avec VGG16 / ResNet50 / InceptionV3

Chaque modèle préentraîné a sa propre taille d'entrée recommandée et sa
propre fonction de preprocessing (voir MODEL_CONFIGS ci-dessous) : il faut
impérativement utiliser la bonne combinaison, sinon les performances du
transfer learning s'effondrent.
"""

from tensorflow.keras import layers, models
import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg16
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet50
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_inception_v3


# Configuration par modèle : taille d'entrée native + fonction de preprocessing associée
MODEL_CONFIGS = {
    "vgg16": {
        "class": VGG16,
        "input_shape": (150, 150, 3),
        "preprocess": preprocess_vgg16,
    },
    "resnet50": {
        "class": ResNet50,
        "input_shape": (224, 224, 3),
        "preprocess": preprocess_resnet50,
    },
    "inception_v3": {
        "class": InceptionV3,
        "input_shape": (299, 299, 3),
        "preprocess": preprocess_inception_v3,
    },
}


def get_data_augmentation() -> tf.keras.Sequential:
    """
    Data augmentation commune à tous les modèles : réduit l'overfitting
    en générant des variations artificielles des images d'entraînement.
    """
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ],
        name="data_augmentation",
    )


def build_cnn_from_scratch(input_shape: tuple = (150, 150, 3)) -> tf.keras.Model:
    """CNN simple entraîné entièrement from scratch, sans transfer learning."""
    model = models.Sequential(
        [
            get_data_augmentation(),
            layers.Rescaling(1.0 / 255),
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(512, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="cnn_from_scratch",
    )
    return model


def build_transfer_model(model_name: str) -> tuple:
    """
    Construit un modèle de transfer learning (base gelée) pour l'architecture
    demandée ('vgg16', 'resnet50' ou 'inception_v3').

    Retourne (model, base_model) : base_model est renvoyé séparément pour
    pouvoir être dégelé ensuite lors du fine-tuning.
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"model_name doit être l'un de {list(MODEL_CONFIGS.keys())}")

    config = MODEL_CONFIGS[model_name]

    base_model = config["class"](
        weights="imagenet",
        include_top=False,
        input_shape=config["input_shape"],
    )
    base_model.trainable = False  # on gèle la base au départ

    inputs = tf.keras.Input(shape=config["input_shape"])
    x = get_data_augmentation()(inputs)
    x = layers.Lambda(config["preprocess"])(x)  # preprocessing spécifique au modèle
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs, name=f"{model_name}_transfer")

    return model, base_model


def unfreeze_for_fine_tuning(base_model: tf.keras.Model, n_layers: int = 4) -> None:
    """
    Dégèle les n_layers dernières couches du modèle de base pour le fine-tuning.
    Les premières couches (motifs génériques : bords, textures) restent gelées ;
    seules les dernières couches (motifs plus spécialisés) sont réajustées.
    """
    base_model.trainable = True
    for layer in base_model.layers[:-n_layers]:
        layer.trainable = False


def compile_model(model: tf.keras.Model, learning_rate: float = 1e-3) -> None:
    """Compile le modèle avec les réglages standards pour une classification binaire."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )