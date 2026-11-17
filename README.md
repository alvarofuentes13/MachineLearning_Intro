# MachineLearning_Intro

Introducción práctica al Machine Learning con Python: manejo de valores incompletos, variables categóricas, diseño de pipelines, validación cruzada, modelos de gradient boosting y detección de errores comunes como el *data leakage*.

Cada script recoge un bloque de conceptos, con el código que he ido escribiendo mientras aprendía.

## Contenido

| Archivo | Tema |
|---|---|
| `missing_values.py` | Tratamiento de valores ausentes: eliminar columnas e imputar con `SimpleImputer` |
| `categorical_values.py` | Variables categóricas: eliminación, *Ordinal Encoding* y *One-Hot Encoding* |
| `pipelines.py` | Pipelines de scikit-learn que unen preprocesado y modelo en un solo objeto |
| `cross_validation.py` | Validación cruzada con `cross_val_score` y comparación de hiperparámetros |
| `gradient_boosting.py` | Gradient boosting con XGBoost y *early stopping* |
| `data/` | Datasets usados en los ejemplos |

## Conceptos que se trabajan

- **Valores ausentes:** comparar distintas estrategias y medir su efecto con el error absoluto medio (MAE).
- **Variables categóricas:** cuándo usar Ordinal Encoding y cuándo One-Hot, y cómo evitar fallos con categorías que solo aparecen en validación (`handle_unknown`).
- **Pipelines:** código más limpio, menos errores y preprocesado aplicado igual en entrenamiento y validación.
- **Validación cruzada:** una estimación más fiable del rendimiento del modelo que una única división entrenamiento/validación.
- **Gradient boosting:** XGBoost, ajuste de `n_estimators` y `learning_rate`, y parada temprana.
- **Data leakage:** cómo detectar fugas de información (*target leakage* y contaminación entrenamiento-validación) que inflan los resultados.

## Tecnologías

- Python 3
- pandas
- scikit-learn
- XGBoost

## Instalación

```bash
git clone https://github.com/alvarofuentes13/MachineLearning_Intro.git
cd MachineLearning_Intro

python -m venv .venv
source .venv/bin/activate      # En Windows: .venv\Scripts\activate

pip install pandas scikit-learn xgboost
```

## Uso

Cada script se ejecuta de forma independiente:

```bash
python missing_values.py
python categorical_values.py
python pipelines.py
python cross_validation.py
python gradient_boosting.py
```

> Los scripts leen los datasets desde la carpeta `data/`. Ejecútalos desde la raíz del repositorio para que las rutas funcionen.

## Notas

- Los ejercicios siguen el curso [Intermediate Machine Learning](https://www.kaggle.com/learn/intermediate-machine-learning) de Kaggle, adaptados para ejecutarse en local.
- En versiones recientes de XGBoost (2.x), `early_stopping_rounds` se pasa al constructor del modelo y no a `fit()`.
- En pandas 3.x las columnas de texto pueden aparecer con dtype `str` en lugar de `object`, así que para detectar variables categóricas conviene usar `select_dtypes(include=["object", "string"])`.

## Autor

**Álvaro Fuentes** — [github.com/alvarofuentes13](https://github.com/alvarofuentes13)