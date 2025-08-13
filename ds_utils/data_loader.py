# src/ds_utiles/data_loader.py
from pathlib import Path
from typing import Any, Dict, Tuple, Optional
import pandas as pd

try:
    from IPython.display import display
except Exception:
    def display(*args, **kwargs):  # no-op fuera de notebook
        for a in args: print(a if isinstance(a, str) else getattr(a, "head", lambda *x: a)())

class DataLoader:
    def load_csv(self, path: str | Path, **kwargs) -> pd.DataFrame:
        return pd.read_csv(path, low_memory=False, **kwargs)

    @staticmethod
    def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = (
            df.columns
              .str.strip()
              .str.lower()
              .str.replace(r"\s+", "_", regex=True)
        )
        return df

    # 1️⃣ Caso local: train.csv y test.csv con target presente
    def load_train_test_data(
        self,
        train_path: str | Path,
        test_path: str | Path,
        target_col: str = "target",
        id_col: Optional[str] = None,
        show_head: bool = False
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, Dict[str, Any]]:

        df_train = self.load_csv(train_path)
        df_test = self.load_csv(test_path)

        if show_head:
            display(df_train.head(3))
            display(df_test.head(3))

        tcol = target_col
        X_train = df_train.drop(columns=[tcol])
        y_train = df_train[tcol]

        X_test = df_test.drop(columns=[tcol])
        y_test = df_test[tcol]

        meta = {
            "target_col": tcol,
            "feature_cols": list(X_train.columns),
            "id_col": id_col,
            "test_ids": df_test[id_col].tolist() if (id_col and id_col in df_test.columns) else None
        }
        return X_train, y_train, X_test, y_test, meta

    # 2️⃣ Caso Kaggle: train.csv con target, test.csv sin target
    def load_kaggle_competition_data(
        self,
        train_path: str | Path,
        test_path: str | Path,
        *,
        target_col: str = "target",
        id_col: Optional[str] = None,
        show_head: bool = False
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, Optional[pd.Series], Dict[str, Any]]:

        df_train = self.load_csv(train_path)
        df_test  = self.load_csv(test_path)

        if show_head:
            try:
                from IPython.display import display
                print("Train head:"); display(df_train.head(5))
                print("Test head:");  display(df_test.head(5))
            except Exception:
                print(df_train.head(5)); print(df_test.head(5))

        # --- validar target en train ---
        tcol = target_col
        if tcol not in df_train.columns:
            raise KeyError(f"Columna target '{tcol}' no está en TRAIN. Columnas: {list(df_train.columns)}")

        # --- detectar ID en test ---
        test_ids = None
        resolved_id_col = None

        if id_col and id_col in df_test.columns:
            resolved_id_col = id_col
            test_ids = df_test[id_col].tolist()
        else:
            first = df_test.columns[0]
            if first.startswith("Unnamed") or first.strip() == "":
                # primera columna sin nombre → usarla como ID
                resolved_id_col = "id"
                df_test = df_test.rename(columns={first: resolved_id_col})
                test_ids = df_test[resolved_id_col].tolist()
            else:
                # alternativa: usar índice como ID (si la primera col es realmente un ID repetido del índice)
                # re-lee usando index_col=0
                df_test = pd.read_csv(test_path, low_memory=False, index_col=0)
                test_ids = df_test.index.tolist()
                resolved_id_col = "id"  # nombre que usaremos en el CSV final

        # --- construir splits (Kaggle: test sin target) ---
        X_train = df_train.drop(columns=[tcol])
        y_train = df_train[tcol]

        if tcol in df_test.columns:
            X_test = df_test.drop(columns=[tcol])
            y_test = df_test[tcol]
        else:
            X_test = df_test
            y_test = None

        # --- eliminar ID de las features en ambos (si quedó como columna) ---
        for c in [resolved_id_col, "Unnamed: 0", "unnamed: 0", ""]:
            if c and c in X_train.columns:
                X_train = X_train.drop(columns=[c])
            if c and c in X_test.columns:
                X_test = X_test.drop(columns=[c])

        # --- alinear columnas (seguridad extra) ---
        X_test = X_test.reindex(columns=X_train.columns)

        # --- meta para submission y reproducibilidad ---
        meta: Dict[str, Any] = {
            "target_col": tcol,
            "feature_cols": list(X_train.columns),
            "id_col": resolved_id_col,
            "test_ids": test_ids,
        }

        return X_train, y_train, X_test, y_test, meta

