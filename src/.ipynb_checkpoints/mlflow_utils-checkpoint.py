import os
import tempfile
from matplotlib import pyplot as plt
import mlflow
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.metrics import (accuracy_score,
                             log_loss,
                             precision_score,
                             recall_score,
                             f1_score,
                             roc_auc_score,
                             confusion_matrix,
                             classification_report,
                             ConfusionMatrixDisplay)
from sklearn.model_selection import cross_val_score 

class MLflowUtils:
    def __init__(self):
        self.uri = None
        self.experiment = None
         
    def init_mlflow(self, uri, experiment: str = "default"):
        mlflow.set_tracking_uri(uri)
        print(f'Create or use experiment: {experiment}')
        mlflow.set_experiment(experiment)

    def log_basic_metrics(self,**metrics):
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))

    def log_confusion_matrix(self,y_true, y_pred, model_name):
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        tmp = tempfile.mkdtemp()
        cm_path = os.path.join(tmp, "confusion_matrix.png")
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path, artifact_path="confusion_matrix")
        plt.close()

    def plot_confusion_matrix(self,y_true, y_pred, title="Matriz de confusión"):
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Real")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center")
        fig.colorbar(im, ax=ax)
        return fig, ax

    def train_and_log_multiclass_model(
        self, model, model_name, X_train, X_test, y_train, y_test, *,
        dataset_name="unknown", params=None, targets_str_col_name="species_name"
    ):
        """
        Entrena un modelo multiclase para dataset_name y registra métricas en MLflow.
        Robusta con y en str o codificada (int) y alinea correctamente etiquetas.
        """
        with mlflow.start_run(run_name=f"{dataset_name} - {model_name}"):
            # ----- Parámetros -----
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("dataset", dataset_name)
            mlflow.log_param("problem_type", "multiclass_classification")
            mlflow.log_param("n_classes", len(np.unique(y_train)))
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))
            mlflow.log_param("n_features", X_train.shape[1])
    
            if params:
                for key, value in params.items():
                    # evita loguear objetos gigantes (p.ej., encoders)
                    mlflow.log_param(str(key), str(value))
    
            # ----- Entrenamiento -----
            model.fit(X_train, y_train)
    
            # ----- Predicciones -----
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    
            # ----- Métricas globales -----
            accuracy = accuracy_score(y_test, y_pred)
            precision_w = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall_w = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1_w = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
            precision_m = precision_score(y_test, y_pred, average='macro', zero_division=0)
            recall_m = recall_score(y_test, y_pred, average='macro', zero_division=0)
            f1_m = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision_weighted", precision_w)
            mlflow.log_metric("recall_weighted", recall_w)
            mlflow.log_metric("f1_weighted", f1_w)
            mlflow.log_metric("precision_macro", precision_m)
            mlflow.log_metric("recall_macro", recall_m)
            mlflow.log_metric("f1_macro", f1_m)
    
            # ----- AUC (multiclase) si hay proba -----
            if y_pred_proba is not None:
                # Alinea columnas de proba con etiquetas usando las clases del modelo
                labels_for_auc = getattr(model, "classes_", None)
                auc_ovr = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', labels=labels_for_auc)
                auc_ovo = roc_auc_score(y_test, y_pred_proba, multi_class='ovo', labels=labels_for_auc)
                mlflow.log_metric("auc_ovr", auc_ovr)
                mlflow.log_metric("auc_ovo", auc_ovo)
    
            # ----- Cross-validation (accuracy) -----
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            mlflow.log_metric("cv_mean", float(cv_scores.mean()))
            mlflow.log_metric("cv_std", float(cv_scores.std()))
    
            # ----- Registrar modelo -----
            mlflow.sklearn.log_model(model, "model")
    
            # ====== Etiquetas y nombres de clase robustos ======
            # 1) Etiquetas base en el mismo orden de las clases del modelo si existen
            if hasattr(model, "classes_"):
                labels = np.array(model.classes_)
            else:
                labels = np.unique(np.concatenate([np.asarray(y_train), np.asarray(y_test), np.asarray(y_pred)]))
    
            # 2) target_names:
            #    - Si pasaste un LabelEncoder en params, usa sus clases (nombres reales)
            #    - Si y es string/objeto, usa los valores de y en el orden de 'labels'
            #    - Si y es numérica, convierte a string para evitar TypeError en classification_report
            le = (params or {}).get("label_encoder", None)
    
            if le is not None and hasattr(le, "classes_"):
                # Asegura que el largo coincida con labels; si no, hace fallback
                if len(le.classes_) == len(labels):
                    name_map = {lab: str(name) for lab, name in zip(labels, le.classes_)}
                    target_names = [name_map.get(lab, str(lab)) for lab in labels]
                else:
                    target_names = [str(lab) for lab in labels]
            else:
                y_kind = np.asarray(y_train).dtype.kind
                if y_kind in ("U", "S", "O"):
                    # y es string/objeto
                    uniq = pd.unique(pd.Series(y_train, name=targets_str_col_name)).tolist()
                    # map al orden de 'labels'
                    name_map = {u: str(u) for u in uniq}
                    target_names = [name_map.get(lab, str(lab)) for lab in labels]
                else:
                    # y es numérica
                    target_names = [str(lab) for lab in labels]
    
            # ----- Matriz de confusión -----
            cm = confusion_matrix(y_test, y_pred, labels=labels)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
            fig = disp.plot(values_format='d').figure_
            plt.title(f'Matriz de Confusión - {model_name} ({dataset_name})')
            plt.tight_layout()
    
            confusion_file = f"{dataset_name}_confusion_matrix_{model_name.replace(' ', '_')}.png"
            fig.savefig(confusion_file)
            mlflow.log_artifact(confusion_file)
            plt.show()
    
            # ----- Reporte de clasificación -----
            report = classification_report(
                y_test, y_pred, labels=labels, target_names=target_names, zero_division=0
            )
            report_file = f"{dataset_name}_classification_report_{model_name.replace(' ', '_')}.txt"
            with open(report_file, "w", encoding="utf-8") as f:
                f.write(report)
            mlflow.log_artifact(report_file)
    
            # ----- Consola -----
            print(f"🌸 {model_name} entrenado en dataset {dataset_name}")
            print(f"📊 Accuracy: {accuracy:.4f}")
            print(f"📊 F1-Score (weighted): {f1_w:.4f}")
            print(f"📊 F1-Score (macro): {f1_m:.4f}")
            if y_pred_proba is not None:
                print(f"📊 AUC (OvR): {auc_ovr:.4f}")
                print(f"📊 AUC (OvO): {auc_ovo:.4f}")
            print(f"📊 CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            print("-" * 60)
    
            return model

