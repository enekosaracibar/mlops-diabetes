# Proyecto MLOps: predicción de la progresión de la diabetes

![CI](https://github.com/enekosaracibar/mlops-diabetes/actions/workflows/ci.yml/badge.svg)

Proyecto final del módulo de MLOps (VASS). Convierte un notebook exploratorio en un repositorio con un flujo MLOps básico: datos → entrenamiento → seguimiento con MLflow → modelo campeón → API → CI/CD.

## 1. Objetivo

Predecir cómo avanza la diabetes de un paciente un año después de la primera medición, a partir de datos como la edad, el índice de masa corporal (IMC) o la tensión arterial.

Es un problema de **regresión**: el resultado es un número continuo.

## 2. Dataset

Se usa el dataset **Diabetes** de scikit-learn: 442 pacientes, 10 variables de entrada y una columna objetivo (`target`).

Se carga con los **valores reales, sin escalar** (`load_diabetes(scaled=False)`). Así los datos se entienden mejor y es más fácil detectar valores imposibles.

| Columna | Significado |
|---|---|
| `age` | Edad en años |
| `sex` | Sexo (1 o 2) |
| `bmi` | Índice de masa corporal |
| `bp` | Tensión arterial media |
| `s1` – `s6` | Seis análisis de sangre (colesterol, LDL, HDL, triglicéridos, glucosa…) |
| `target` | Progresión de la enfermedad un año después (lo que se predice) |

## 3. Estructura del repositorio

```
mlops-diabetes/
├── .github/workflows/ci.yml   # CI/CD con GitHub Actions
├── app/main.py                # API con FastAPI
├── config/config.py           # Configuración central (rutas, columnas, MLflow, umbral)
├── examples/                  # Pacientes de ejemplo para probar la API
├── notebooks/exploracion.ipynb  # Notebook exploratorio
├── src/
│   ├── data/
│   │   ├── extract_data.py    # Extracción → data/raw/
│   │   ├── validate_data.py   # Validación del esquema
│   │   └── clean_data.py      # Limpieza → data/processed/
│   └── models/
│       ├── train.py           # Entrenamiento + seguimiento en MLflow
│       ├── register_model.py  # Registro del mejor modelo
│       └── evaluate_model.py  # Quality gate y alias champion
├── run_pipeline.py            # Ejecuta todo el pipeline en orden
└── requirements.txt           # Dependencias con versiones fijadas
```

Las carpetas `data/`, `mlruns/` y el archivo `mlflow.db` no se suben a Git: se generan al ejecutar el pipeline.

## 4. Instalación

Requisitos: **Python 3.12** y **Git**.

```bash
git clone https://github.com/enekosaracibar/mlops-diabetes.git
cd mlops-diabetes
python -m venv .venv
```

Activar el entorno virtual:

- Windows (PowerShell): `.venv\Scripts\Activate.ps1`
- Linux / macOS: `source .venv/bin/activate`

Instalar las dependencias:

```bash
pip install -r requirements.txt
```

Todos los comandos siguientes se ejecutan desde la carpeta raíz del proyecto.

## 5. Pipeline de datos

Tres scripts, en este orden:

```bash
python -m src.data.extract_data    # Guarda los datos en bruto en data/raw/diabetes.csv
python -m src.data.validate_data   # Comprueba columnas, objetivo y tipos
python -m src.data.clean_data      # Limpia y guarda en data/processed/diabetes_clean.csv
```

- **Validación:** comprueba que el dataset no está vacío, que existen la columna objetivo y todas las de entrada, y que todas son numéricas. Si algo falla, el script termina con error y el pipeline se para.
- **Limpieza:** convierte todo a número, elimina duplicados, marca como nulos los valores imposibles (sexo distinto de 1 o 2, o valores ≤ 0 en el resto), elimina las filas sin objetivo y rellena los nulos con la mediana (el sexo, con el valor más frecuente).

Estos pasos se ven primero en el notebook (`notebooks/exploracion.ipynb`, secciones 5 a 7).

## 6. Entrenamiento y registro del modelo

```bash
python -m src.models.train            # Entrena 3 modelos y los registra en MLflow
python -m src.models.register_model   # Registra el mejor (menor RMSE) en el Model Registry
python -m src.models.evaluate_model   # Aplica el quality gate y asigna el alias champion
```

- **Entrenamiento:** compara LinearRegression, RandomForest y GradientBoosting (80 % entrenamiento, 20 % prueba). Cada modelo es un *run* de MLflow con sus parámetros, sus métricas (RMSE, MAE, R²) y el modelo guardado como artefacto.
- **MLflow** guarda la información en `mlflow.db` (SQLite) y los modelos en `mlruns/`.
- **Registro:** el run con menor RMSE se registra como una nueva versión de `diabetes-model`.

Para ver los resultados en la interfaz web de MLflow:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000 --workers 1
```

Y abrir http://127.0.0.1:5000

Resultados obtenidos (conjunto de prueba):

| Modelo | RMSE | MAE | R² |
|---|---|---|---|
| LinearRegression | 53,85 | 42,79 | 0,453 |
| RandomForest | 54,35 | 44,08 | 0,443 |
| **GradientBoosting** | **53,73** | 44,52 | **0,455** |

## 7. Cómo se elige el modelo campeón

`evaluate_model.py` evalúa la última versión registrada con el conjunto de prueba y aplica un **quality gate**:

1. **RMSE < 60** → aprobado. Si no, se rechaza y no recibe el alias.
2. Además, solo se convierte en **`champion`** si mejora (o iguala) al champion actual. Así una versión peor nunca sustituye a una mejor.

El resultado se guarda como etiquetas de la versión (`rmse`, `r2`, `quality_gate`).

**Por qué el umbral es 60:**

| Referencia | RMSE |
|---|---|
| Predecir siempre la media del entrenamiento | ≈ 73 |
| **Umbral** | **60** |
| Modelos entrenados | 53,7 – 54,4 |

- Exige que el modelo sea claramente mejor que predecir la media (al menos un 18 % menos de error).
- Deja un margen sobre los resultados actuales, para que pequeñas variaciones (por ejemplo, al actualizar librerías) no rechacen un modelo válido.

El umbral está en `config/config.py` (`RMSE_THRESHOLD`).

## 8. Ejecutar todo el pipeline

Los seis pasos anteriores, en orden y con un solo comando:

```bash
python run_pipeline.py
```

## 9. API

La API carga al arrancar el modelo con el alias **`champion`** desde el Model Registry de MLflow (`models:/diabetes-model@champion`). Necesita haber ejecutado antes el pipeline.

```bash
uvicorn app.main:app --port 8000
```

| Método | Endpoint | Qué hace |
|---|---|---|
| GET | `/health` | Comprueba que el servicio está vivo |
| GET | `/model-info` | Devuelve el modelo servido: nombre, versión, alias, run y métricas |
| POST | `/predict` | Recibe los datos de un paciente en JSON y devuelve la predicción y la versión del modelo |

## 10. Probar la API

**Opción 1: navegador.** Abrir http://127.0.0.1:8000/docs (documentación automática). En cada endpoint: *Try it out* → *Execute*. `/predict` trae un paciente de ejemplo ya relleno.

**Opción 2: terminal.** En Windows PowerShell hay que escribir `curl.exe` en lugar de `curl`.

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/model-info
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "@examples/paciente.json"
```

Respuestas esperadas:

- `/health` → `{"status": "ok", "model_loaded": true}`
- `/predict` → `{"prediction": 237.36, "model_version": "1"}`

## 11. Gestión de errores

| Caso | Código | Cómo se gestiona |
|---|---|---|
| Petición correcta | **200** | Devuelve la predicción |
| Falta un campo, tipo incorrecto o valor imposible | **422** | Pydantic valida el JSON antes de llegar al modelo |
| Modelo no cargado | **503** | Si no se pudo cargar el champion al arrancar, `/predict` y `/model-info` responden "Modelo no cargado" |
| Fallo al predecir | **500** | Un `try/except` captura cualquier error del modelo y responde "Error al generar la predicción" |

Para probar el **422**:

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "@examples/paciente_sin_bmi.json"
```

Para probar el **503**, arrancar la API buscando un alias que no existe:

- Windows (PowerShell): `$env:MODEL_ALIAS = "no-existe"` y después `uvicorn app.main:app --port 8000`
- Linux / macOS: `MODEL_ALIAS=no-existe uvicorn app.main:app --port 8000`

## 12. CI/CD con GitHub Actions

El workflow `.github/workflows/ci.yml` se ejecuta automáticamente en cada **push a `main` o `master`** (y también a mano desde la pestaña *Actions*). En una máquina Linux limpia:

1. Descarga el código e instala Python 3.12 y las dependencias.
2. Ejecuta el pipeline en orden: extracción → validación → limpieza → entrenamiento → registro → evaluación.
3. Si un paso falla, los siguientes no se ejecutan.
4. Guarda `mlflow.db` y `mlruns/` como *artifact* descargable (`mlflow-resultados`), porque la máquina se borra al terminar.

El resultado de cada ejecución se ve en la pestaña **Actions** del repositorio.

## 13. Decisiones técnicas

- **MLflow con SQLite** (`mlflow.db`): desde MLflow 3.7 es el almacenamiento por defecto, y el Model Registry necesita una base de datos.
- **Alias en lugar de *stages***: los stages (Staging/Production) están obsoletos en MLflow; el alias `champion` permite cambiar de versión sin tocar la API.
- **Versiones fijadas** en `requirements.txt`, para obtener los mismos resultados en local y en GitHub Actions.
- **Formato *skops*** para guardar los modelos (por defecto en MLflow 3.17), declarando como confiable el tipo interno de los modelos de árboles.