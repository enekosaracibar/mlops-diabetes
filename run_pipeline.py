"""Orquestación: ejecuta el pipeline completo en orden."""

import logging

from src.data import clean_data, extract_data, validate_data
from src.models import evaluate_model, register_model, train

logger = logging.getLogger(__name__)

# Pasos del pipeline, en el orden en que se ejecutan
STEPS = [
    ("Extracción", extract_data.main),
    ("Validación", validate_data.main),
    ("Limpieza", clean_data.main),
    ("Entrenamiento", train.main),
    ("Registro", register_model.main),
    ("Evaluación y champion", evaluate_model.main),
]


def main() -> None:
    for i, (name, step) in enumerate(STEPS, start=1):
        logger.info("=== Paso %d/%d: %s ===", i, len(STEPS), name)
        step()
    logger.info("=== Pipeline completado ===")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()