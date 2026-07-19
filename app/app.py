"""
Interface web Gradio pour tester le classifieur chat vs chien.
Lancer en local :  python app/app.py
Déployé sur Hugging Face Spaces, ce même fichier sert de point d'entrée
(Hugging Face détecte automatiquement app.py + requirements.txt).
"""

import numpy as np
import gradio as gr
import tensorflow as tf
from PIL import Image

MODEL_PATH = "cat_dog_classifier.keras"

# IMPORTANT : doit correspondre exactement à la taille utilisée à l'entraînement.
# 150x150 pour un CNN from scratch ou VGG16, 224x224 pour ResNet50, 299x299 pour InceptionV3.
IMG_SIZE = (224, 224)

model = tf.keras.models.load_model(MODEL_PATH)


def predire(image: Image.Image) -> str:
    if image is None:
        return "Merci d'uploader une image."

    img = image.convert("RGB").resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    # Note : si le modèle inclut déjà la couche de preprocessing (Lambda) en interne
    # (cas des modèles transfer learning construits via build_transfer_model),
    # on passe ici les pixels bruts [0, 255] ; le modèle s'occupe du reste.
    prediction = model.predict(img_array, verbose=0)[0][0]

    if prediction > 0.5:
        return f"🐶 Chien (confiance : {prediction:.1%})"
    else:
        return f"🐱 Chat (confiance : {(1 - prediction):.1%})"


interface = gr.Interface(
    fn=predire,
    inputs=gr.Image(type="pil", label="Photo à classifier"),
    outputs=gr.Textbox(label="Résultat"),
    title="🐱🐶 Classification Chat vs Chien",
    description=(
        "Upload une image et le modèle (CNN + Transfer Learning) prédit "
        "s'il s'agit d'un chat ou d'un chien."
    ),
    examples=None,
)

if __name__ == "__main__":
    interface.launch()