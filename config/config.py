"""Configuración central del proyecto: rutas, nombre del modelo, alias y umbral.

Cada paso del proyecto añade aquí sus valores para no repetirlos en los scripts.
"""

from pathlib import Path

# Carpeta raíz del proyecto (la que contiene config/, src/, data/...)
ROOT_DIR = Path(__file__).resolve().parent.parent

# Rutas de los datos
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "diabetes.csv"

PROCESSED_DATA_PATH = ROOT_DIR / "data" / "processed" / "diabetes_clean.csv"

# Columnas del dataset
FEATURES = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
TARGET = "target"