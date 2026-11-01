"""Configuración central del proyecto: rutas, nombre del modelo, alias y umbral.

Cada paso del proyecto añade aquí sus valores para no repetirlos en los scripts.
"""

from pathlib import Path

# Carpeta raíz del proyecto (la que contiene config/, src/, data/...)
ROOT_DIR = Path(__file__).resolve().parent.parent

# Rutas de los datos
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "diabetes.csv"