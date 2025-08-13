# ds_utils/submissions.py  (o dentro de data_loader.py si preferís)
from pathlib import Path
import pandas as pd

def make_submission(test_csv, y_pred, 
                    out_path, 
                    id_source=None,
                    id_col_name="ID", 
                    target_col_name="TARGET"):
    """
    Genera un archivo de submission con columnas y orden exactos como en test.csv.
    
    test_csv         → Ruta al archivo test.csv oficial.
    y_pred           → Array/serie con predicciones.
    out_path         → Ruta donde guardar el submission.
    id_source        → Nombre de la columna con IDs en test.csv.
                       Si None, toma la primera columna.
    id_col_name      → Nombre EXACTO que Kaggle espera para la columna de IDs.
    target_col_name  → Nombre EXACTO que Kaggle espera para la columna de predicciones.
    """
    test_df = pd.read_csv(test_csv)
    
    if id_source:
        ids = test_df[id_source]
    else:
        ids = test_df.iloc[:, 0]
    
    if len(ids) != len(y_pred):
        raise ValueError(f"Longitud de IDs ({len(ids)}) y predicciones ({len(y_pred)}) no coincide.")
    
    sub = pd.DataFrame({
        id_col_name: ids,
        target_col_name: y_pred
    })
    
    sub.to_csv(out_path, index=False)
    return sub

def compare_first_column(csv1, csv2):
    """
    Compara los valores de la primera columna entre dos archivos CSV.
    Imprime si son iguales o muestra las diferencias.
    """
    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2)
    col1 = df1.columns[0]
    col2 = df2.columns[0]
    vals1 = df1[col1].tolist()
    vals2 = df2[col2].tolist()
    if vals1 == vals2:
        print(f"La primera columna de ambos archivos son iguales.")
    else:
        print(f"La primera columna son diferentes.")
        # Mostrar diferencias
        set1 = set(vals1)
        set2 = set(vals2)
        only_in_1 = set1 - set2
        only_in_2 = set2 - set1
        if only_in_1:
            print(f"Valores solo en {csv1}: {only_in_1}")
        if only_in_2:
            print(f"Valores solo en {csv2}: {only_in_2}")


        
   