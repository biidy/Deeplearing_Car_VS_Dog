"""
Script d'entraînement du classifieur chat vs chien.

Exemples :
    python src/train.py --model scratch --epochs 15
    python src/train.py --model resnet50 --epochs 10 --fine-tune --fine-tune-epochs 10
"""

import argparse

from data_preprocessing import charger_datasets
from model import (
    MODEL_CONFIGS,
    build_cnn_from_scratch,
    build_transfer_model,
    compile_model,
    unfreeze_for_fine_tuning,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Entraîne un classifieur chat vs chien")
    parser.add_argument(
        "--model",
        type=str,
        default="resnet50",
        choices=["scratch", "vgg16", "resnet50", "inception_v3"],
        help="Architecture à utiliser",
    )
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=15, help="Epochs d'entraînement de la tête")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--fine-tune",
        action="store_true",
        help="Active une seconde phase de fine-tuning après l'entraînement de la tête",
    )
    parser.add_argument("--fine-tune-epochs", type=int, default=10)
    parser.add_argument("--fine-tune-layers", type=int, default=4)
    parser.add_argument("--output", type=str, default="cat_dog_classifier.keras")
    return parser.parse_args()


def main():
    args = parse_args()

    # Détermine la taille d'image attendue par le modèle choisi
    if args.model == "scratch":
        img_size = (150, 150)
    else:
        img_size = MODEL_CONFIGS[args.model]["input_shape"][:2]

    print(f"Chargement des données (taille image : {img_size}) ...")
    train_ds, val_ds = charger_datasets(
        data_dir=args.data_dir, img_size=img_size, batch_size=args.batch_size
    )

    # Construction du modèle
    base_model = None
    if args.model == "scratch":
        model = build_cnn_from_scratch(input_shape=img_size + (3,))
    else:
        model, base_model = build_transfer_model(args.model)

    compile_model(model, learning_rate=1e-3)
    model.summary()

    print(f"\n--- Entraînement de la tête ({args.epochs} epochs) ---")
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)

    # Fine-tuning optionnel (uniquement pertinent pour le transfer learning)
    if args.fine_tune and base_model is not None:
        print(f"\n--- Fine-tuning ({args.fine_tune_epochs} epochs) ---")
        unfreeze_for_fine_tuning(base_model, n_layers=args.fine_tune_layers)
        # Learning rate très bas pour ne pas détruire les features préentraînées
        compile_model(model, learning_rate=1e-5)
        model.fit(train_ds, validation_data=val_ds, epochs=args.fine_tune_epochs)

    model.save(args.output)
    print(f"\nModèle sauvegardé : {args.output}")


if __name__ == "__main__":
    main()