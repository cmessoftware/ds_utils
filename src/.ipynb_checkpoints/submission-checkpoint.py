import pandas as pd
from IPython.display import display

__all__ = ['SubmissionCreator']

class SubmissionCreator:
    def __init__(self, final_model):
        self.final_model = final_model
    
    def create_submission_file(self, X_train_full, y_train_full, X_test_full, customer_ids, filename="submission.csv"):
        """
        Entrena el modelo final con todos los datos de entrenamiento, genera predicciones
        de probabilidad en el conjunto de prueba y guarda el archivo de submission.

        Args:
            X_train_full (DataFrame): El DataFrame completo de características de entrenamiento.
            y_train_full (Series): La Serie completa del objetivo de entrenamiento.
            X_test_full (DataFrame): El DataFrame de características de prueba.
            customer_ids (Series): La Serie de customerID para el archivo de submission.
            filename (str): El nombre del archivo CSV de salida.
        """
        print("Entrenando el modelo final con todos los datos de entrenamiento...")
        self.final_model.fit(X_train_full, y_train_full)
        print("Modelo final entrenado.")

        print("Generando predicciones de probabilidad sobre el conjunto de prueba...")
        test_probabilities = self.final_model.predict_proba(X_test_full)[:, 1]

        print(f"Creando el archivo de submission '{filename}'...")
        submission_df = pd.DataFrame({
            'customerID': customer_ids,
            'Churn': test_probabilities
        })

        submission_df.to_csv(filename, index=False)

        print(f"Archivo '{filename}' generado exitosamente.")
        print("Primeras 5 filas del archivo de submission:")
        display(submission_df.head())
        return submission_df