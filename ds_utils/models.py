#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de Modelos para Predicción de Target - Versión Estable

Versión sin emojis optimizada para máxima compatibilidad
Incluye TargetPredictor y funciones de optimización de hiperparámetros
"""

import warnings

warnings.filterwarnings("ignore")

import time
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Import display for Jupyter environments
try:
    from IPython.display import display
except ImportError:
    # Fallback for non-Jupyter environments
    def display(obj, **kwargs):
        print(obj)
        return None


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from tqdm.auto import tqdm
from datetime import datetime
from joblib import parallel_backend
from scipy.stats import rv_continuous


# Importar tqdm para barra de progreso
# try:
#     from tqdm import tqdm

#     TQDM_AVAILABLE = True
# except ImportError:
#     TQDM_AVAILABLE = False
#     print("Warning: tqdm no disponible - usando progress básico")


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Cast seguro
        if "TotalCharges" in X.columns:
            X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")
        if "tenure" in X.columns:
            X["avg_monthly"] = X["TotalCharges"] / (X["tenure"] + 1)
            X["tenure_group"] = pd.cut(
                X["tenure"],
                bins=[0, 6, 24, 60, 100],
                labels=["0-6", "7-24", "25-60", "60+"],
                include_lowest=True,
            )
            X["new_customer"] = (X["tenure"] < 6).astype(int)
        if "MonthlyCharges" in X.columns:
            X["high_monthly_charge"] = (X["MonthlyCharges"] > 80).astype(int)
        
        return X


class TargetPredictor:
    """
    Predictor de Target - Versión Estable

    Clase principal para el modelado de predicción de abandono de clientes
    Incluye preprocesamiento, entrenamiento y evaluación de modelos
    """

    def __init__(self, random_state=42):
        """
        Inicializar el predictor de Target

        Args:
            random_state (int): Semilla para reproducibilidad
        """
        self.random_state = random_state
        self.preprocessor = None
        self.models = {}
        self.results = {}

        print(f"🗺️  TargetPredictor inicializado con random_state={random_state}")

    def _step_name(self, step):
        # Devuelve nombre legible del step (clase o función)
        if isinstance(step, tuple) and len(step) == 2:
            name, obj = step
        else:
            name, obj = None, step
        try:
            cls = obj.__class__.__name__
        except Exception:
            cls = str(type(obj))
        # Para FunctionTransformer, muestra la función
        if isinstance(obj, FunctionTransformer) and obj.func is not None:
            return f"{cls}({getattr(obj.func, '__name__', str(obj.func))})"
        return cls

    def _describe_transformer(self, trf):
        """
        Devuelve (transformer_type, steps_str) para mostrar en tabla.
        - Si es Pipeline: lista de steps.
        - Si es ColumnTransformer: nombre y cantidad de sub-bloques.
        - Si es otro estimador: su clase.
        """
        if isinstance(trf, Pipeline):
            steps = [self._step_name(s) for s in trf.steps]
            return ("Pipeline", " -> ".join(steps))
        if isinstance(trf, ColumnTransformer):
            # raro aquí (anidado), pero lo contemplamos
            return ("ColumnTransformer", f"{len(trf.transformers)} sub-bloques")
        if trf in ("drop", "passthrough"):
            return (str(trf), "")
        try:
            return (trf.__class__.__name__, "")
        except Exception:
            return (str(type(trf)), "")

    def _column_transformers_to_df(self, models_dict):
        """
        models_dict: dict[str, ColumnTransformer]
          Ej: {'Logistic_Regression': <ColumnTransformer ...>, ...}
        """
        rows = []
        for model_name, coltr in models_dict.items():
            if not isinstance(coltr, ColumnTransformer):
                # Si vino un Pipeline completo, intenta extraer el paso 'preprocessor'
                if isinstance(coltr, Pipeline) and "preprocessor" in dict(coltr.steps):
                    coltr = dict(coltr.steps)["preprocessor"]
                else:
                    # No es ColumnTransformer; registra y sigue
                    rows.append(
                        {
                            "model": model_name,
                            "block": "",
                            "transformer_type": coltr.__class__.__name__,
                            "steps": "",
                            "columns": "",
                        }
                    )
                    continue

            # coltr.transformers es lista de tuplas (name, transformer, columns)
            for name, trf, cols in coltr.transformers:
                trf_type, steps_desc = self._describe_transformer(trf)

                # columnas pueden venir como lista, slice, 'drop', 'passthrough', etc.
                if isinstance(cols, (list, tuple)):
                    cols_str = ", ".join(map(str, cols))
                else:
                    cols_str = str(cols)

                rows.append(
                    {
                        "model": model_name,
                        "block": name,
                        "transformer_type": trf_type,
                        "steps": steps_desc,
                        "columns": cols_str,
                    }
                )

        df = pd.DataFrame(
            rows, columns=["model", "block", "transformer_type", "steps", "columns"]
        )
        return df

    def _normalize_service_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reemplaza valores como 'No internet service' por 'No' en columnas de servicios opcionales.
        """
        df = df.copy()

        # Columnas que pueden tener 'No internet service'
        internet_service_cols = [
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]

        # Columnas que pueden tener 'No phone service'
        phone_service_cols = ["MultipleLines"]

        # Normalizar servicios de internet
        for col in internet_service_cols:
            if col in df.columns:
                df[col] = df[col].replace("No internet service", "No")

        # Normalizar servicios de teléfono
        for col in phone_service_cols:
            if col in df.columns:
                df[col] = df[col].replace("No phone service", "No")

        return df

    def _transform_yes_no(self, X):
        """
        Transforma columnas con valores 'Yes'/'No' a 1/0.

        Args:
            X: Serie de pandas o numpy array con valores 'Yes'/'No'.

        Returns:
            Array transformado a 1/0.
        """
        if hasattr(X, "map"):
            # Es pandas Series
            return X.map({"Yes": 1, "No": 0}).fillna(0).astype(int)
        else:
            # Es numpy array
            import numpy as np

            result = np.where(X == "Yes", 1, np.where(X == "No", 0, 0))
            return result.astype(int)

    def _transform_yes_no_df(self, X):
        """
        Transforma DataFrame con columnas 'Yes'/'No' a 1/0.

        Args:
            X: DataFrame con columnas 'Yes'/'No' o numpy array 2D.

        Returns:
            DataFrame o array transformado a 1/0.
        """
        if hasattr(X, "apply"):
            # Es pandas DataFrame
            return X.apply(self._transform_yes_no)
        else:
            # Es numpy array 2D
            import numpy as np

            if len(X.shape) == 2:
                # Array 2D - aplicar transformación a cada columna
                result = np.zeros_like(X, dtype=int)
                for i in range(X.shape[1]):
                    result[:, i] = self._transform_yes_no(X[:, i])
                return result
            else:
                # Array 1D
                return self._transform_yes_no(X)

    def _transform_male_female(self, X):
        """
        Transforma columnas con valores 'Male'/'Female' a 1/0.
        Args:
            X: Serie de pandas o numpy array con valores 'Male'/'Female'.

        Returns:
            Array transformado a 1/0.
        """
        if hasattr(X, "map"):
            # Es pandas Series
            return X.map({"Male": 1, "Female": 0}).fillna(0).astype(int)
        else:
            # Es numpy array
            import numpy as np

            result = np.where(X == "Male", 1, np.where(X == "Female", 0, 0))
            return result.astype(int)

    def inspect_transformed_columns(
        self,
        X_original: pd.DataFrame,
        columns: list,
        fit=True,
        current_preprocessor=None,
    ):
        """
        Muestra comparativa entre las columnas originales y sus transformaciones numéricas.

        Args:
            X_original (pd.DataFrame): Dataset original sin transformar.
            columns (list): Lista de columnas a inspeccionar.
            fit (bool): Si True, hace fit_transform; si False, solo transform.
        """
        print(f"🔍 Inspeccionando transformación de columnas: {columns}")

        if current_preprocessor is not None:
            self.preprocessor = current_preprocessor

        print(f"Usando preprocesador: {self.preprocessor}")

        # Asegurarse de usar solo las columnas seleccionadas
        X_subset = X_original.copy()

        # Fit o solo transform (según contexto)
        if fit:
            X_transformed = self.preprocessor.fit_transform(X_subset)
        else:
            X_transformed = self.preprocessor.transform(X_subset)

        # Obtener nombres de columnas transformadas con verificación de dimensiones
        try:
            print(f"🔍 Debug: X_transformed shape: {X_transformed.shape}")

            # Obtener las columnas de cada tipo de manera segura
            num_cols = []
            bin_cols = []
            cat_cols = []

            # Iterar sobre los transformers disponibles
            for name, transformer, cols in self.preprocessor.transformers_:
                if name == "num":
                    num_cols = list(cols)
                elif name == "bin":
                    bin_cols = list(cols)
                elif name == "cat":
                    cat_cols = list(cols)

            print(
                f"🔍 Debug: num_cols={len(num_cols)}, bin_cols={len(bin_cols)}, cat_cols={len(cat_cols)}"
            )

            # Para categorical features, obtener nombres después de one-hot encoding
            cat_names = []
            if cat_cols and "cat" in self.preprocessor.named_transformers_:
                cat_transformer = self.preprocessor.named_transformers_["cat"]
                if (
                    hasattr(cat_transformer, "named_steps")
                    and "encoder" in cat_transformer.named_steps
                ):
                    cat_encoder = cat_transformer.named_steps["encoder"]
                    if hasattr(cat_encoder, "get_feature_names_out"):
                        cat_names = list(cat_encoder.get_feature_names_out(cat_cols))
                        print(f"🔍 Debug: cat_names length: {len(cat_names)}")
                elif hasattr(cat_transformer, "get_feature_names_out"):
                    cat_names = list(cat_transformer.get_feature_names_out(cat_cols))
                    print(f"🔍 Debug: cat_names length: {len(cat_names)}")

            final_cols = num_cols + bin_cols + cat_names
            print(f"🔍 Debug: final_cols length: {len(final_cols)}")

            # Verificar que las dimensiones coincidan
            if len(final_cols) != X_transformed.shape[1]:
                print(
                    f"⚠️ Mismatch: final_cols={len(final_cols)}, X_transformed.shape[1]={X_transformed.shape[1]}"
                )
                raise ValueError("Dimensiones no coinciden")

        except Exception as e:
            print(f"⚠️ No se pudieron obtener nombres de columnas: {e}")
            # Fallback con nombres genéricos basado en la forma real
            final_cols = [f"feature_{i}" for i in range(X_transformed.shape[1])]
            print(f"🔧 Usando fallback con {len(final_cols)} columnas genéricas")

        # Convertir a DataFrame
        df_transformed = pd.DataFrame(X_transformed, columns=final_cols)

        # Buscar todas las columnas transformadas que derivan de las columnas originales seleccionadas
        matched_cols = []
        for col in columns:
            # Coincidencias exactas o one-hot
            matched = [
                c for c in df_transformed.columns if col == c or c.startswith(col + "_")
            ]
            matched_cols.extend(matched)

        # Mostrar comparación lado a lado
        display("\n🗂️  Valores originales:")
        print(X_subset[columns].head())

        print("⚙️  Valores transformados:")
        if matched_cols:
            print(df_transformed[matched_cols].head())
        else:
            print(
                "⚠️ No se encontraron columnas coincidentes en los datos transformados"
            )
            print(
                f"Columnas disponibles: {list(df_transformed.columns[:10])}..."
            )  # Mostrar primeras 10

    def create_preprocessor_random_forest(self, X: pd.DataFrame) -> ColumnTransformer:
        """
        Justificación:
            Random Forest no necesita escalado.
            Soporta bien variables numéricas y categóricas dummificadas.
            Se recomienda eliminar variables irrelevantes pero mantener su codificación completa.
        """
        X = self._normalize_service_values(X)

        id_cols = [col for col in X.columns if "id" in col.lower()]
        numeric_features = (
            X.select_dtypes(include=["int64", "float64"])
            .columns.difference(id_cols)
            .tolist()
        )
        object_features = (
            X.select_dtypes(include="object").columns.difference(id_cols).tolist()
        )

        binary_features = [
            col
            for col in object_features
            if set(X[col].dropna().unique()) <= {"Yes", "No"}
        ]
        categorical_features = [
            col for col in object_features if col not in binary_features
        ]

        yes_no_transformer = FunctionTransformer(self._transform_yes_no, validate=False)

        return ColumnTransformer(
            [
                ("num", SimpleImputer(strategy="median"), numeric_features),
                (
                    "bin",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("transformer", yes_no_transformer),
                        ]
                    ),
                    binary_features,
                ),
                (
                    "cat",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            (
                                "encoder",
                                OneHotEncoder(drop="first", handle_unknown="ignore"),
                            ),
                        ]
                    ),
                    categorical_features,
                ),
            ]
        )

    def create_preprocessor_svm(self, X: pd.DataFrame) -> ColumnTransformer:
        """
        Numéricas: imputación (mediana) + StandardScaler.
        Categóricas: imputación (más frecuente) + One-Hot.
        Para SVC(kernel='rbf'), usar salida densa (sparse=False) para evitar densificar luego.
        Si preferís LinearSVC, podés dejar sparse=True (más liviano en memoria).
        """
        if X is None or X.empty:
            raise ValueError("X no puede ser None o vacío")

        X = self._normalize_service_values(X)

        num_cols: List[str] = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols: List[str] = X.select_dtypes(exclude=[np.number]).columns.tolist()

        numeric = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric, num_cols),
                ("cat", categorical, cat_cols),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

        return preprocessor

    def create_preprocessor_xgboost(self, X: pd.DataFrame) -> ColumnTransformer:
        raise NotImplementedError("Aún no implementado")

    def create_preprocessor_naive_bayes(self, X: pd.DataFrame) -> ColumnTransformer:
        """
        Justificación:
            Naive Bayes clásico (GaussianNB) no se lleva bien con variables dummificadas dispersas.
            Recomendado: LabelEncoder para categóricas multiclase, y mantener las numéricas tal cual.
            Alternativa práctica: convertir multiclase a ordinal.
        """
        from sklearn.preprocessing import OrdinalEncoder

        X = self._normalize_service_values(X)

        id_cols = [col for col in X.columns if "id" in col.lower()]
        numeric_features = (
            X.select_dtypes(include=["int64", "float64"])
            .columns.difference(id_cols)
            .tolist()
        )
        object_features = (
            X.select_dtypes(include="object").columns.difference(id_cols).tolist()
        )

        binary_features = [
            col
            for col in object_features
            if set(X[col].dropna().unique()) <= {"Yes", "No"}
        ]
        categorical_features = [
            col for col in object_features if col not in binary_features
        ]

        yes_no_transformer = FunctionTransformer(self._transform_yes_no, validate=False)

        return ColumnTransformer(
            [
                (
                    "num",
                    Pipeline([("imputer", SimpleImputer(strategy="mean"))]),
                    numeric_features,
                ),
                (
                    "bin",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("transformer", yes_no_transformer),
                        ]
                    ),
                    binary_features,
                ),
                (
                    "cat",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            (
                                "encoder",
                                OrdinalEncoder(
                                    handle_unknown="use_encoded_value", unknown_value=-1
                                ),
                            ),
                        ]
                    ),
                    categorical_features,
                ),
            ]
        )

    def create_preprocessor_knn(self, X: pd.DataFrame) -> ColumnTransformer:
        """
        Justificación:
            KNN es muy sensible a la escala, así que todo debe estar escalado.
            Una opción es usar OneHotEncoder(sparse_output=False) + StandardScaler para todo.
        """
        X = self._normalize_service_values(X)

        id_cols = [col for col in X.columns if "id" in col.lower()]
        numeric_features = (
            X.select_dtypes(include=["int64", "float64"])
            .columns.difference(id_cols)
            .tolist()
        )
        object_features = (
            X.select_dtypes(include="object").columns.difference(id_cols).tolist()
        )

        binary_features = [
            col
            for col in object_features
            if set(X[col].dropna().unique()) <= {"Yes", "No"}
        ]
        categorical_features = [
            col for col in object_features if col not in binary_features
        ]

        yes_no_transformer = FunctionTransformer(self._transform_yes_no, validate=False)

        return ColumnTransformer(
            [
                (
                    "num",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    numeric_features,
                ),
                (
                    "bin",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("transformer", yes_no_transformer),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    binary_features,
                ),
                (
                    "cat",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            (
                                "encoder",
                                OneHotEncoder(drop="first", sparse_output=False),
                            ),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    categorical_features,
                ),
            ]
        )

    def create_preprocessor_logistic_regression(
        self, X: pd.DataFrame
    ) -> ColumnTransformer:
        """
        Justificación:
            Le va mejor con variables escaladas.
            Soporta bien variables dummificadas.
            Evitar muchas columnas poco informativas.
        """
        X = self._normalize_service_values(X)

        id_cols = [col for col in X.columns if "id" in col.lower()]
        numeric_features = (
            X.select_dtypes(include=["int64", "float64"])
            .columns.difference(id_cols)
            .tolist()
        )
        object_features = (
            X.select_dtypes(include="object").columns.difference(id_cols).tolist()
        )

        binary_features = [
            col
            for col in object_features
            if set(X[col].dropna().unique()) <= {"Yes", "No"}
        ]
        categorical_features = [
            col for col in object_features if col not in binary_features
        ]

        yes_no_transformer = FunctionTransformer(self._transform_yes_no, validate=False)

        return ColumnTransformer(
            [
                (
                    "num",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    numeric_features,
                ),
                (
                    "bin",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("transformer", yes_no_transformer),
                        ]
                    ),
                    binary_features,
                ),
                (
                    "cat",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            (
                                "encoder",
                                OneHotEncoder(drop="first", handle_unknown="ignore"),
                            ),
                        ]
                    ),
                    categorical_features,
                ),
            ]
        )

    def create_preprocesor_gradient_boosting(
        self, X: pd.DataFrame
    ) -> ColumnTransformer:
        """
        Justificación:
              No requiere escalado (los árboles son insensibles a escala)
              Es sensible a dummie ero puede manejar high-cardinality
              Requiere imputación	en el manejo de nuklos, pero algunos GB modernos lo hacen internamente
              GB no necesita one-hot; puede usar label encoding u ordinal encoding (o directamente si es CatBoost o LGBM con soporte nativo)

        Parametros para GradientBoostingClassifier :
            | Parámetro            | Significado                                                        | Justificación común                                                                                      |
            |----------------------|--------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
            | `n_estimators=150`   | Número de árboles en el ensamble                                   | Más árboles mejoran el rendimiento hasta cierto punto. 150 es un valor razonable.                        |
            | `learning_rate=0.1`  | Cuánto contribuye cada nuevo árbol al modelo final                 | Valor clásico (default). Reduce riesgo de overfitting si se aumenta `n_estimators`.                      |
            | `max_depth=6`        | Profundidad máxima de cada árbol                                   | Árboles más profundos capturan más patrones, pero con riesgo de overfitting.                             |
            | `random_state=42`    | Semilla para reproducibilidad                                      | Fija los resultados entre ejecuciones.                                                                   |
            | `subsample=1.0`      | Proporción de muestras para entrenar cada árbol                    | Si < 1.0 introduce aleatoriedad que puede reducir overfitting. Ej: `0.8` se usa en stochastic boosting.  |
            | `min_samples_split=2`| Mínimo número de muestras para dividir un nodo                     | Aumentar este valor hace el modelo más conservador.                                                      |
            | `min_samples_leaf=1` | Mínimo número de muestras en una hoja terminal                     | Útil para suavizar el modelo y evitar hojas pequeñas que sobreajusten.                                   |
            | `max_features=None`  | Número máximo de features evaluadas al dividir un nodo             | Limitarlo (`'sqrt'`, `'log2'`, número o fracción) puede reducir el overfitting.                          |
            | `loss='log_loss'`    | Función de pérdida a optimizar                                     | `'log_loss'` (default) para clasificación binaria, también disponible: `'exponential'`.                  |
            | `criterion='friedman_mse'` | Función para medir la calidad de una división                | `'friedman_mse'` es robusta y adecuada para boosting.                                                    |
            | `init=None`          | Modelo inicial antes de aplicar boosting                           | Se puede usar un modelo base, pero `None` usa predicción constante inicial.                              |
            | `warm_start=False`   | Continuar entrenamiento desde una previa ejecución                 | Útil para ajustar más árboles sin volver a entrenar desde cero.                                          |


        """
        X = self._normalize_service_values(X)

        id_cols = [col for col in X.columns if "id" in col.lower()]
        numeric_features = (
            X.select_dtypes(include=["int64", "float64"])
            .columns.difference(id_cols)
            .tolist()
        )
        object_features = (
            X.select_dtypes(include="object").columns.difference(id_cols).tolist()
        )

        binary_features = [
            col
            for col in object_features
            if set(X[col].dropna().unique()) <= {"Yes", "No"}
        ]
        categorical_features = [
            col for col in object_features if col not in binary_features
        ]

        yes_no_transformer = FunctionTransformer(self._transform_yes_no, validate=False)

        return ColumnTransformer(
            [
                (
                    "num",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            # No scaler
                        ]
                    ),
                    numeric_features,
                ),
                (
                    "bin",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("transformer", yes_no_transformer),
                        ]
                    ),
                    binary_features,
                ),
                (
                    "cat",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            (
                                "encoder",
                                OneHotEncoder(drop="first", handle_unknown="ignore"),
                            ),
                        ]
                    ),
                    categorical_features,
                ),
            ]
        )

    def create_preprocessor(self, X_train):
        """
        Crear el preprocesador para las características - VERSIÓN MEJORADA CON BALANCEO

        Args:
            X_train: Dataset de entrenamiento

        Returns:
            ColumnTransformer: Preprocesador configurado
        """
        print("🗺️  Creando preprocesador mejorado...")

        # Remover columnas problemáticas
        X_clean = X_train.copy()
        if "Target" in X_clean.columns:
            X_clean = X_clean.drop("Target", axis=1)
        if "customerID" in X_clean.columns:
            X_clean = X_clean.drop("customerID", axis=1)

        # Identificar tipos de columnas de manera más robusta
        categorical_features = []
        numerical_features = []

        for col in X_clean.columns:
            if X_clean[col].dtype == "object":
                categorical_features.append(col)
            else:
                numerical_features.append(col)

        print(
            f"🗺️  Características categóricas ({len(categorical_features)}): {categorical_features}"
        )
        print(
            f"🗺️  Características numéricas ({len(numerical_features)}): {numerical_features}"
        )

        # Crear transformadores mejorados con imputación
        numerical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                (
                    "encoder",
                    OneHotEncoder(
                        drop="first", handle_unknown="ignore", sparse_output=False
                    ),
                ),
            ]
        )

        # Crear preprocesador mejorado
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_pipeline, numerical_features),
                ("cat", categorical_pipeline, categorical_features),
            ],
            remainder="drop",  # Eliminar cualquier columna no especificada
        )

        self.preprocessor = preprocessor
        print("✅ Preprocesador mejorado creado exitosamente")
        return preprocessor

    # --- 2) Helper para elegir preprocesador por modelo
    def _pick_preprocessor(self, model_name: str, X: pd.DataFrame) -> ColumnTransformer:
        if model_name == "Logistic_Regression":
            return self.create_preprocessor_logistic_regression(X)
        if model_name == "KNN":
            return self.create_preprocessor_knn(X)
        if model_name == "Naive_Bayes":
            return self.create_preprocessor_naive_bayes(X)
        if model_name == "Random_Forest":
            return self.create_preprocessor_random_forest(X)
        if model_name == "Gradient_":
            return self.create_preprocesor_gradient_boosting(X)
        # Fallback (tu genérico)
        return self.create_preprocessor(X)

    
    def create_pipeline(self, classifier):
        """
        Arma un pipeline completo con:
        - Ingeniería de features opcional
        - Preprocesamiento numérico y categórico
        - Modelo final inyectado como parámetro
        """
    
        # 🔧 Columnas numéricas y categóricas (ajustar según tus datos)
        num_features = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'avg_monthly']
        bin_features = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 
                        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                        'TechSupport', 'StreamingTV', 'StreamingMovies']
    
        cat_features = ['gender', 'InternetService', 'Contract', 'PaymentMethod']
    
      
        def _normalize_and_engineer(Z: pd.DataFrame) -> pd.DataFrame:
            Z = self._normalize_service_values(Z.copy())
            fe = FeatureEngineer()
            return fe.transform(Z)

    
        feature_engineering = FunctionTransformer(_normalize_and_engineer)
    
        # ⚙️ Preprocesamiento
        num_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
    
        bin_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent"))
            # No scaler ni encoding, ya están en formato 0/1 o Yes/No
        ])
    
        cat_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))
        ])
    
        preprocessor = ColumnTransformer(transformers=[
            ("num", num_pipeline, num_features),
            ("bin", "passthrough", bin_features),  # se asume que ya están preprocesadas como 0/1
            ("cat", cat_pipeline, cat_features)
        ])
    
        # 🧪 Pipeline completo
        pipeline = Pipeline(steps=[
            ("features", feature_engineering),
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ])
    
        return pipeline

   
    def create_models(self, X: pd.DataFrame):
        """
        Crear diccionario de modelos con pipelines optimizados por algoritmo.
        Requiere X con todas las columnas (features sin target).
        """
        print("🗺️  Creando modelos...")

        # Paso común: ingeniería de features + normalización de strings de servicios
        def _normalize_and_engineer(Z: pd.DataFrame) -> pd.DataFrame:
            Z = self._normalize_service_values(Z.copy())
            fe = FeatureEngineer()
            return fe.transform(Z)

        feat_step = (
            "features",
            FunctionTransformer(_normalize_and_engineer, validate=False),
        )

        # Preprocesadores específicos (se calculan con las columnas tras feature engineering)
        X_for_schema = _normalize_and_engineer(X)

        preprocessors = {
            "Logistic_Regression": self._pick_preprocessor(
                "Logistic_Regression", X_for_schema
            ),
            "Random_Forest": self._pick_preprocessor("Random_Forest", X_for_schema),
            "Naive_Bayes": self._pick_preprocessor("Naive_Bayes", X_for_schema),
            "KNN": self._pick_preprocessor("KNN", X_for_schema),
            "Gradient_Boosting": self._pick_preprocessor(
                "Gradient_Boosting", X_for_schema
            ),
        }

        df_view = self._column_transformers_to_df(preprocessors)
        if running_context == "jupyter_notebook":
            display(df_view)  # en notebook
        else:
            print(df_view.to_string(index=False))  # en script

        print("✅ Preprocessors were configured successfully")

        models = {
            "Logistic_Regression": Pipeline(
                [
                    feat_step,
                    ("preprocessor", preprocessors["Logistic_Regression"]),
                    (
                        "classifier",
                        LogisticRegression(
                            random_state=self.random_state,
                            max_iter=1000,
                            class_weight="balanced",  # MEJORA: Balanceo de clases --> IMPORTANTE PARA TELCO Target que tiene alto desbalanceo entre No Target / Target
                        ),
                    ),
                ]
            ),
            "Random_Forest": Pipeline(
                [
                    feat_step,
                    ("preprocessor", preprocessors["Random_Forest"]),
                    (
                        "classifier",
                        RandomForestClassifier(
                            random_state=self.random_state,
                            n_estimators=200,
                            max_depth=10,
                            min_samples_split=10,
                            min_samples_leaf=5,
                            class_weight="balanced",  # MEJORA: Balanceo de clases
                        ),
                    ),
                ]
            ),
            "Gradient_Boosting": Pipeline(
                [
                    feat_step,
                    (
                        "preprocessor",
                        preprocessors["Random_Forest"],
                    ),  # Usar mismo preprocessor que RF
                    (
                        "classifier",
                        GradientBoostingClassifier(
                            random_state=self.random_state,
                            n_estimators=150,
                            learning_rate=0.1,
                            max_depth=6,
                        ),
                    ),
                ]
            ),
            "Naive_Bayes": Pipeline(
                [
                    feat_step,
                    ("preprocessor", preprocessors["Naive_Bayes"]),
                    ("classifier", GaussianNB()),
                ]
            ),
            "KNN": Pipeline(
                [
                    feat_step,
                    ("preprocessor", preprocessors["KNN"]),
                    (
                        "classifier",
                        KNeighborsClassifier(n_neighbors=7, weights="distance"),
                    ),
                ]
            ),
        }

        self.models = models
        print(f"✅ {len(models)} modelos creados:")
        for name in models.keys():
            print(f"  - {name}")
        return models

    def train_models(self, X_train, y_train):
        """
        Entrenar todos los modelos con barra de progreso

        Args:
            X_train: Características de entrenamiento
            y_train: Variable objetivo de entrenamiento
        """
        print("🗺️  Iniciando entrenamiento de modelos...")
        print(f"🗺️  Total de modelos a entrenar: {len(self.models)}")

        # Variables para tracking de tiempo
        start_time = time.time()
        model_times = []
        successful_models = 0
        failed_models = 0

        # Configurar progress bar - VERSION SIMPLE Y CONFIABLE
        models_list = list(self.models.items())

        # if TQDM_AVAILABLE:
        #     progress_iterator = tqdm(
        #         models_list,
        #         desc="Entrenando modelos",
        #         unit="modelo",
        #         ascii=True,  # ASCII para máxima compatibilidad
        #         ncols=70,  # Ancho fijo
        #         leave=True,  # Dejar visible para debug
        #     )
        #     use_tqdm = True
        # else:
        #     progress_iterator = models_list
        #     use_tqdm = False

        try:
            for i, (name, model) in enumerate(models_list, 1):
                model_start_time = time.time()

                try:
                    print(f"[{i}/{len(self.models)}] Entrenando {name}...")

                    # Entrenar el modelo
                    model.fit(X_train, y_train)

                    # Calcular tiempo de entrenamiento
                    model_time = time.time() - model_start_time
                    model_times.append(model_time)
                    successful_models += 1

                    print(f"✅ {name} entrenado en {model_time:.1f}s")

                except Exception as e:
                    failed_models += 1
                    model_time = time.time() - model_start_time
                    model_times.append(model_time)

                    print(f"❌Error entrenando {name}: {e}")

            # Resumen final - SIMPLE Y CLARO
            total_time = time.time() - start_time
            avg_time = sum(model_times) / len(model_times) if model_times else 0

            # Limpiar salida
            # if use_tqdm:
            #     time.sleep(0.1)  # Pequeña pausa

            print()  # Línea en blanco
            print("=" * 50)
            print("ENTRENAMIENTO COMPLETADO")
            print("=" * 50)
            print(f"Tiempo total: {total_time:.1f}s")
            print(f"Tiempo promedio por modelo: {avg_time:.1f}s")
            print(f"✅ Modelos exitosos: {successful_models}")
            if failed_models > 0:
                print(f"❌ Modelos fallidos: {failed_models}")
            print(f"Tasa de éxito: {(successful_models/len(self.models)*100):.1f}%")
            print("=" * 50)
        except Exception as e:
            print(f"❌Error general en entrenamiento: {e}")
            print(f"      Tipo de error: {type(e).__name__}")
            if hasattr(e, "__cause__") and e.__cause__:
                print(f"      Causa original: {e.__cause__}")

    def evaluate_models(self, X_test, y_test):
        """
        Evaluar todos los modelos entrenados

        Args:
            X_test: Características de prueba
            y_test: Variable objetivo de prueba

        Returns:
            dict: Resultados de evaluación
        """
        print("🗺️  Evaluando modelos...")

        results = {}

        for name, model in self.models.items():
            try:
                print(f"🗺️  Evaluando {name}...")

                # Predicciones
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]

                # Métricas
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(
                    y_test, y_pred, average="weighted", zero_division=0
                )
                recall = recall_score(
                    y_test, y_pred, average="weighted", zero_division=0
                )
                f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
                roc_auc = roc_auc_score(y_test, y_pred_proba)

                results[name] = {
                    "Accuracy": accuracy,
                    "Precision": precision,
                    "Recall": recall,
                    "F1_Score": f1,
                    "ROC_AUC": roc_auc,
                }

                print(f"✅ {name} evaluado - ROC AUC: {roc_auc:.4f}")

            except Exception as e:
                print(f"❌Error evaluando {name}: {e}")
                results[name] = {
                    "Accuracy": 0.0,
                    "Precision": 0.0,
                    "Recall": 0.0,
                    "F1_Score": 0.0,
                    "ROC_AUC": 0.0,
                }

        self.results = results
        print("✅ Evaluación completada!")
        return results

    def get_best_model(self, metric="ROC_AUC", results=None):
        """
        Obtener el mejor modelo basado en una métrica

        Args:
            metric (str): Métrica para seleccionar el mejor modelo
            results (dict): Resultados de evaluación (opcional)

        Returns:
            tuple: (nombre_modelo, modelo)
        """
        if results is None:
            results = self.results

        print(f"🗺️  Seleccionando mejor modelo por {metric}...")

        best_score = -1
        best_model_name = None

        for name, metrics in results.items():
            if metrics[metric] > best_score:
                best_score = metrics[metric]
                best_model_name = name

        if best_model_name is None:
            print("❌No se encontró ningún modelo válido")
            return None, None

        best_model = self.models[best_model_name]

        print(f"✅ Mejor modelo: {best_model_name} ({metric}: {best_score:.4f})")

        return best_model_name, best_model

    def predict_proba(self, X):
        """Predicciones de probabilidad"""
        X_processed = self.preprocessor.transform(X)
        return self.model.predict_proba(X_processed)

    def predict(self, X):
        """Predicciones binarias"""
        X_processed = self.preprocessor.transform(X)
        return self.model.predict(X_processed)

    def generate_model_report(self, X_test, y_test):
        """
        Generar reporte detallado de modelos

        Args:
            X_test: Características de prueba
            y_test: Variable objetivo de prueba
        """
        print("🗺️  Generando reporte de modelos...")

        print("\n" + "=" * 60)
        print("REPORTE DETALLADO DE MODELOS")
        print("=" * 60)

        for name, model in self.models.items():
            try:
                print(f"\nMODELO: {name}")
                print("-" * 40)

                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]

                # Métricas básicas
                # print('Call accuracy_score()')
                accuracy = accuracy_score(y_test, y_pred)
                # print('Call precision_score()')
                precision = precision_score(
                    y_test, y_pred, average="weighted", zero_division=0
                )
                # print('Call recall_score()')
                recall = recall_score(
                    y_test, y_pred, average="weighted", zero_division=0
                )
                # print('Call f1_score()')
                f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
                # print('Call roc_auc_score()')
                roc_auc = roc_auc_score(y_test, y_pred_proba)

                print(f"Accuracy:  {accuracy:.4f}")
                print(f"Precision: {precision:.4f}")
                print(f"Recall:    {recall:.4f}")
                print(f"F1-Score:  {f1:.4f}")
                print(f"ROC AUC:   {roc_auc:.4f}")

                # Matriz de confusión
                self.show_confusion_matrix(y_test, y_pred, name)

            except Exception as e:
                print(f"❌Error generando reporte para {name}: {e}")

        print("\n" + "=" * 60)
        print("✅ Reporte completado")

    def show_confusion_matrix(self, y_test, y_pred, model_name):
        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nMatriz de Confusión para {model_name}")
        print(f"   TN: {cm[0,0]:4d} | FP: {cm[0,1]:4d}")
        print(f"   FN: {cm[1,0]:4d} | TP: {cm[1,1]:4d}")

    def prepare_data(self, df):
        """
        Preparar datos separando características (X) y variable objetivo (y)

        Args:
            df: DataFrame con datos completos incluyendo target

        Returns:
            tuple: (X, y) - características y variable objetivo
        """
        print("🗺️  Preparando datos...")

        # Verificar que el DataFrame no esté vacío
        if df is None or df.empty:
            raise ValueError("El DataFrame no puede ser None o vacío")

        # Hacer una copia para no modificar el original
        df_copy = df.copy()

        # Identificar la columna objetivo
        target_col = "Target"
        if target_col not in df_copy.columns:
            raise ValueError(
                f"Columna objetivo '{target_col}' no encontrada en el DataFrame"
            )

        # Separar características y objetivo
        y = df_copy[target_col]

        # Remover columnas que no son características
        columns_to_drop = [target_col]
        if "customerID" in df_copy.columns:
            columns_to_drop.append("customerID")

        X = df_copy.drop(columns=columns_to_drop, axis=1)

        # Mapear variable objetivo si es necesario
        y_mapped = self.map_target(y)

        print(f"✅ Datos preparados: X{X.shape}, y{y_mapped.shape}")
        print(f"📊 Características: {list(X.columns)}")
        print(f"📊 Distribución objetivo: {y_mapped.value_counts().to_dict()}")

        return X, y_mapped

    def create_advanced_features(self, df):
        """
        MEJORA: Función de ingeniería de características avanzada para Target prediction

        Crea características más predictivas basadas en el análisis de datos:
        - Características financieras: ratios, detección de altos cargos
        - Características de servicios: conteo de servicios, servicios de protección
        - Características de contrato: identificación de contratos cortos/largos
        - Características demográficas: procesamiento mejorado

        Args:
            df (pd.DataFrame): Dataset original

        Returns:
            pd.DataFrame: Dataset con características mejoradas
        """
        df_improvement = df.copy()

        # Eliminar columna Target si existe (para evitar data leakage en test)
        if "Target" in df_improvement.columns:
            df_improvement = df_improvement.drop("Target", axis=1)

        # Características financieras
        if "MonthlyCharges" in df.columns and "TotalCharges" in df.columns:
            # Convertir TotalCharges a numérico
            df_improvement["TotalCharges"] = pd.to_numeric(
                df_improvement["TotalCharges"], errors="coerce"
            )
            df_improvement["TotalCharges"] = df_improvement["TotalCharges"].fillna(
                df_improvement["TotalCharges"].median()
            )

            # Nuevas características financieras
            df_improvement["Charge_Per_Month_Ratio"] = df_improvement[
                "TotalCharges"
            ] / (df_improvement["tenure"] + 1)
            df_improvement["High_Monthly_Charges"] = (
                df_improvement["MonthlyCharges"]
                > df_improvement["MonthlyCharges"].quantile(0.75)
            ).astype(int)
            df_improvement["High_Total_Charges"] = (
                df_improvement["TotalCharges"]
                > df_improvement["TotalCharges"].quantile(0.75)
            ).astype(int)

        # Características de servicios
        service_cols = [
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]

        available_services = [col for col in service_cols if col in df.columns]
        if available_services:
            # Contar servicios activos
            df_improvement["Total_Services"] = 0
            for col in available_services:
                if col == "PhoneService":
                    df_improvement["Total_Services"] += (
                        df_improvement[col] == "Yes"
                    ).astype(int)
                elif col == "InternetService":
                    df_improvement["Total_Services"] += (
                        df_improvement[col] != "No"
                    ).astype(int)
                else:
                    df_improvement["Total_Services"] += (
                        df_improvement[col] == "Yes"
                    ).astype(int)

            # Servicios de protección
            protection_services = [
                "OnlineSecurity",
                "OnlineBackup",
                "DeviceProtection",
                "TechSupport",
            ]
            available_protection = [
                col for col in protection_services if col in df.columns
            ]
            if available_protection:
                df_improvement["Protection_Services"] = 0
                for col in available_protection:
                    df_improvement["Protection_Services"] += (
                        df_improvement[col] == "Yes"
                    ).astype(int)

        # Características demográficas
        if "SeniorCitizen" in df.columns:
            df_improvement["SeniorCitizen_Int"] = df_improvement[
                "SeniorCitizen"
            ].astype(int)

        # Características de contrato
        if "Contract" in df.columns:
            df_improvement["Short_Contract"] = (
                df_improvement["Contract"] == "Month-to-month"
            ).astype(int)
            df_improvement["Long_Contract"] = (
                df_improvement["Contract"] == "Two year"
            ).astype(int)

        if "PaymentMethod" in df.columns:
            df_improvement["Electronic_Payment"] = (
                df_improvement["PaymentMethod"] == "Electronic check"
            ).astype(int)

        return df_improvement

    def map_target(self, y):
        """
        Mapear variable objetivo a formato numérico

        Args:
            y: Variable objetivo (puede ser 'Yes'/'No' o 0/1)

        Returns:
            Series/Array: Variable objetivo mapeada a 0/1
        """
        print("🗺️  Mapeando variable objetivo...")

        # Convertir a pandas Series si es numpy array
        if isinstance(y, np.ndarray):
            y = pd.Series(y)
            print("🔄 Convertido numpy array a pandas Series")

        # Verificar si es texto o numérico
        if hasattr(y, "dtype") and y.dtype == "object":
            # Si es texto, mapear a números
            if hasattr(y, "map"):
                y_mapped = y.map({"No": 0, "Yes": 1})
                print("✅ Variable objetivo mapeada: 'No'->0, 'Yes'->1")
            else:
                # Fallback para arrays
                y_mapped = np.where(y == "Yes", 1, 0)
                y_mapped = pd.Series(y_mapped)
                print("✅ Variable objetivo mapeada con np.where: 'No'->0, 'Yes'->1")
        else:
            # Si ya es numérico, mantener como está
            y_mapped = y
            print("✅ Variable objetivo ya es numérica")

        return y_mapped



# def hyperparameter_tuning(model, param_grid, X_train, y_train, cv=5, scoring="roc_auc", mode="auto", n_iter=20):
#     """
#     Optimiza hiperparámetros con GridSearchCV o RandomizedSearchCV.
#     Muestra progreso con tqdm.
#     """
#     print("🧪 Iniciando optimización de hiperparámetros...")
#     print(f"   - CV: {cv} | Scoring: {scoring} | Modo: {mode}")

#     # Validar claves
#     valid_keys = set(model.get_params().keys())
#     unknown = [k for k in param_grid if k not in valid_keys]
#     if unknown:
#         raise ValueError(f"Claves inválidas en la grilla: {unknown}")

#     # Detectar modo si es auto
#     if mode == "auto":
#         mode = "quick" if any(isinstance(v, rv_continuous) for v in param_grid.values()) else "accuracy"

#     use_random = mode == "quick"
#     cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

#     # Construcción del objeto de búsqueda
#     search = RandomizedSearchCV(
#         estimator=model,
#         param_distributions=param_grid,
#         n_iter=n_iter,
#         cv=cv_splitter,
#         scoring=scoring,
#         n_jobs=-1,
#         verbose=0,
#         random_state=42,
#         error_score=np.nan,
#     ) if use_random else GridSearchCV(
#         estimator=model,
#         param_grid=param_grid,
#         cv=cv_splitter,
#         scoring=scoring,
#         n_jobs=-1,
#         verbose=0,
#         error_score=np.nan,
#     )

#     print("⏳ Ejecutando búsqueda...")
#     start = datetime.now()
#     print(f"🕒 Inicio: {start.strftime('%H:%M:%S')}")

#     # Mostrar barra de progreso
#     with parallel_backend("loky"):
#         tqdm_search = tqdm(desc="🔍 Optimizando..", total=n_iter if use_random else None)
#         search.fit(X_train, y_train)
#         tqdm_search.update()

#     end = datetime.now()
#     print(f"✅ Finalizado en {end - start} (Fin: {end.strftime('%H:%M:%S')})")

#     # Mostrar resultados
#     print(f"🎯 Mejor score CV: {search.best_score_:.4f}")
#     print(f"🔍 Mejores parámetros: {search.best_params_}")

#     return search


def hyperparameter_tuning(model, param_grid, X_train, y_train, cv=5, scoring="roc_auc", mode="auto", n_iter=20):
    """
    Optimización robusta de hiperparámetros para cualquier modelo de clasificación.
    
    Parámetros:
    - model: pipeline con el estimador ('classifier')
    - param_grid: grilla o distribución de hiperparámetros
    - X_train, y_train: datos de entrenamiento
    - cv: cantidad de folds de validación cruzada
    - scoring: métrica (default 'roc_auc')
    - mode: 'quick' (random), 'accuracy' (grid), o 'auto'
    - n_iter: solo usado en modo 'quick'
    
    Retorna:
    - Objeto GridSearchCV o RandomizedSearchCV entrenado
    """

    print("🧪 Iniciando optimización de hiperparámetros...")
    print(f"   - CV: {cv} | Scoring: {scoring} | Modo: {mode}")

    #Preprocesar como lo hace el pipeline
    X_train_proc = self._normalize_service_values(X_train.copy())
    X_train_proc = FeatureEngineer().transform(X_train_proc)

    # Validar claves del pipeline
    valid_keys = set(model.get_params().keys())
    unknown = [k for k in param_grid if k not in valid_keys]
    if unknown:
        print("❌ Claves inválidas detectadas:")
        for k in unknown:
            print(f"   - {k}")
        return None

    # Determinar modo real si está en 'auto'
    if mode == "auto":
        mode = "quick" if any(isinstance(v, rv_continuous) for v in param_grid.values()) else "accuracy"

    print(f"🚀 Tipo de búsqueda: {'RandomizedSearchCV' if mode == 'quick' else 'GridSearchCV'}")

    # CV estratificado
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    # Definir el tipo de búsqueda
    if mode == "quick":
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv_splitter,
            scoring=scoring,
            n_jobs=-1,
            verbose=1,
            random_state=42,
            error_score=np.nan,
        )
    else:
        search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv_splitter,
            scoring=scoring,
            n_jobs=-1,
            verbose=1,
            error_score=np.nan,
        )

    # Ejecutar
    print("⏳ Ejecutando búsqueda...")
    start_time = datetime.now()
    print(f"🕒 Inicio: {start_time.strftime('%H:%M:%S')}")

    try:
        search.fit(X_train, y_train)
    except Exception as e:
        print(f"❌ Error durante el ajuste: {e}")
        return None

    end_time = datetime.now()
    duration = end_time - start_time
    print(f"✅ Búsqueda completada en {duration} (Fin: {end_time.strftime('%H:%M:%S')})")

    # Mostrar resultados
    try:
        print(f"🎯 Mejor score CV: {search.best_score_:.4f}")
        print(f"🔍 Mejores parámetros: {search.best_params_}")
    except Exception as e:
        print(f"⚠️ Error al mostrar resultados: {e}")
        return None

    return search


# Función de utilidad para mostrar información del módulo
def show_module_info():
    """
    Mostrar información del módulo
    """
    print("=" * 60)
    print("MODULO MODELS - VERSION ESTABLE")
    print("=" * 60)
    print("Versión sin emojis optimizada para máxima compatibilidad")
    print("Incluye:")
    print("   - TargetPredictor - Clase principal para modelado")
    print("   - hyperparameter_tuning - Optimización de hiperparámetros")
    print("   - Funciones de reporte y evaluación")
    print("=" * 60)


if __name__ == "__main__":
    show_module_info()
