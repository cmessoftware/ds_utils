"""
Módulo para el cálculo y visualización de métricas de evaluación.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class MetricsCalculator:
    """
    Clase para calcular y visualizar métricas de evaluación.
    """
    
    def __init__(self):
        self.metrics_history = []
    
    def calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calcula métricas básicas de clasificación.
        
        Args:
            y_true (np.ndarray): Valores reales
            y_pred (np.ndarray): Predicciones
            y_pred_proba (np.ndarray, optional): Probabilidades predichas
            
        Returns:
            Dict[str, float]: Diccionario con las métricas
        """
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted')
        }
        
        # Agregar métricas que requieren probabilidades
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            metrics['average_precision'] = average_precision_score(y_true, y_pred_proba)
        
        return metrics
    
    def calculate_confusion_matrix_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calcula métricas derivadas de la matriz de confusión.
        
        Args:
            y_true (np.ndarray): Valores reales
            y_pred (np.ndarray): Predicciones
            
        Returns:
            Dict[str, float]: Métricas de la matriz de confusión
        """
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Para clasificación binaria
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            
            metrics = {
                'true_positives': tp,
                'true_negatives': tn,
                'false_positives': fp,
                'false_negatives': fn,
                'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,  # Recall
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
                'positive_predictive_value': tp / (tp + fp) if (tp + fp) > 0 else 0,  # Precision
                'negative_predictive_value': tn / (tn + fn) if (tn + fn) > 0 else 0,
                'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
                'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0
            }
        else:
            # Para clasificación multiclase
            metrics = {
                'confusion_matrix': cm.tolist()
            }
        
        return metrics
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             class_names: Optional[List[str]] = None,
                             normalize: bool = False) -> None:
        """
        Plotea la matriz de confusión.
        
        Args:
            y_true (np.ndarray): Valores reales
            y_pred (np.ndarray): Predicciones
            class_names (List[str], optional): Nombres de las clases
            normalize (bool): Si normalizar la matriz
        """
        
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
            title = 'Matriz de Confusión Normalizada'
        else:
            fmt = 'd'
            title = 'Matriz de Confusión'
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=class_names or ['Clase 0', 'Clase 1'],
                   yticklabels=class_names or ['Clase 0', 'Clase 1'])
        
        plt.title(title)
        plt.xlabel('Predicción')
        plt.ylabel('Valor Real')
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                      model_name: str = "Modelo") -> float:
        """
        Plotea la curva ROC.
        
        Args:
            y_true (np.ndarray): Valores reales
            y_pred_proba (np.ndarray): Probabilidades predichas
            model_name (str): Nombre del modelo
            
        Returns:
            float: Área bajo la curva ROC
        """
        
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        auc_score = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Línea Base (AUC = 0.5)')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Tasa de Falsos Positivos')
        plt.ylabel('Tasa de Verdaderos Positivos')
        plt.title('Curva ROC')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return auc_score
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                   model_name: str = "Modelo") -> float:
        """
        Plotea la curva Precision-Recall.
        
        Args:
            y_true (np.ndarray): Valores reales
            y_pred_proba (np.ndarray): Probabilidades predichas
            model_name (str): Nombre del modelo
            
        Returns:
            float: Área bajo la curva Precision-Recall
        """
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        ap_score = average_precision_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2, label=f'{model_name} (AP = {ap_score:.3f})')
        
        # Línea base (proporción de positivos)
        baseline = np.sum(y_true) / len(y_true)
        plt.axhline(y=baseline, color='k', linestyle='--', alpha=0.5, 
                   label=f'Línea Base (AP = {baseline:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Curva Precision-Recall')
        plt.legend(loc="lower left")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return ap_score
    
    def find_optimal_threshold(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                              metric: str = 'f1') -> Tuple[float, float]:
        """
        Encuentra el umbral óptimo para una métrica específica.
        
        Args:
            y_true (np.ndarray): Valores reales
            y_pred_proba (np.ndarray): Probabilidades predichas
            metric (str): Métrica a optimizar ('f1', 'precision', 'recall', 'accuracy')
            
        Returns:
            Tuple[float, float]: Umbral óptimo y valor de la métrica
        """
        
        thresholds = np.linspace(0, 1, 101)
        scores = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_true, y_pred)
            elif metric == 'precision':
                score = precision_score(y_true, y_pred, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_true, y_pred)
            elif metric == 'accuracy':
                score = accuracy_score(y_true, y_pred)
            else:
                raise ValueError(f"Métrica no soportada: {metric}")
            
            scores.append(score)
        
        # Encontrar el umbral óptimo
        optimal_idx = np.argmax(scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_score = scores[optimal_idx]
        
        # Plotear
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, scores, linewidth=2)
        plt.axvline(x=optimal_threshold, color='r', linestyle='--', 
                   label=f'Umbral Óptimo = {optimal_threshold:.3f}')
        plt.axhline(y=optimal_score, color='r', linestyle='--', alpha=0.5)
        
        plt.xlabel('Umbral')
        plt.ylabel(f'{metric.capitalize()} Score')
        plt.title(f'Optimización de Umbral para {metric.capitalize()}')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        print(f"🎯 Umbral óptimo para {metric}: {optimal_threshold:.3f}")
        print(f"📊 {metric.capitalize()} score: {optimal_score:.4f}")
        
        return optimal_threshold, optimal_score
    
    def compare_models(self, results: Dict[str, Dict[str, float]]) -> None:
        """
        Compara múltiples modelos visualmente.
        
        Args:
            results (Dict[str, Dict[str, float]]): Resultados de múltiples modelos
        """
        
        if not results:
            print("❌ No hay resultados para comparar")
            return
        
        # Convertir a DataFrame
        df = pd.DataFrame(results).T
        
        # Número de métricas
        n_metrics = len(df.columns)
        
        # Configurar subplots
        fig, axes = plt.subplots(2, (n_metrics + 1) // 2, figsize=(15, 10))
        if n_metrics == 1:
            axes = [axes]
        elif n_metrics <= 2:
            axes = axes.flatten()[:n_metrics]
        else:
            axes = axes.flatten()
        
        # Plotear cada métrica
        for i, metric in enumerate(df.columns):
            ax = axes[i]
            
            # Ordenar modelos por métrica
            sorted_data = df[metric].sort_values(ascending=True)
            
            # Crear gráfico de barras horizontal
            bars = ax.barh(range(len(sorted_data)), sorted_data.values, 
                          color=plt.cm.viridis(np.linspace(0, 1, len(sorted_data))))
            
            # Configurar ejes
            ax.set_yticks(range(len(sorted_data)))
            ax.set_yticklabels(sorted_data.index)
            ax.set_xlabel(metric.replace('_', ' ').title())
            ax.set_title(f'Comparación - {metric.replace("_", " ").title()}')
            
            # Añadir valores en las barras
            for j, (bar, value) in enumerate(zip(bars, sorted_data.values)):
                ax.text(value + 0.01, j, f'{value:.3f}', 
                       va='center', ha='left', fontsize=9)
            
            ax.set_xlim(0, 1.1)
            ax.grid(axis='x', alpha=0.3)
        
        # Ocultar subplots extra
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def generate_detailed_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                               y_pred_proba: Optional[np.ndarray] = None,
                               class_names: Optional[List[str]] = None,
                               model_name: str = "Modelo") -> Dict[str, any]:
        """
        Genera un reporte detallado de métricas.
        
        Args:
            y_true (np.ndarray): Valores reales
            y_pred (np.ndarray): Predicciones
            y_pred_proba (np.ndarray, optional): Probabilidades predichas
            class_names (List[str], optional): Nombres de las clases
            model_name (str): Nombre del modelo
            
        Returns:
            Dict[str, any]: Reporte completo
        """
        
        print(f"📋 REPORTE DETALLADO - {model_name}")
        print("=" * 60)
        
        # Métricas básicas
        print('Call calculate_basic_metrics')
        basic_metrics = self.calculate_basic_metrics(y_true, y_pred, y_pred_proba)
        print(f"\n📊 Métricas Básicas:")
        for metric, value in basic_metrics.items():
            print(f"   {metric.replace('_', ' ').title()}: {value:.4f}")
        
        # Métricas de matriz de confusión
        cm_metrics = self.calculate_confusion_matrix_metrics(y_true, y_pred)
        if 'true_positives' in cm_metrics:  # Clasificación binaria
            print(f"\n🎯 Métricas de Matriz de Confusión:")
            print(f"   Verdaderos Positivos: {cm_metrics['true_positives']}")
            print(f"   Verdaderos Negativos: {cm_metrics['true_negatives']}")
            print(f"   Falsos Positivos: {cm_metrics['false_positives']}")
            print(f"   Falsos Negativos: {cm_metrics['false_negatives']}")
            print(f"   Sensibilidad (Recall): {cm_metrics['sensitivity']:.4f}")
            print(f"   Especificidad: {cm_metrics['specificity']:.4f}")
        
        # Reporte de clasificación de sklearn
        print(f"\n📈 Reporte de Clasificación:")
        print(classification_report(y_true, y_pred, target_names=class_names))
        
        # Visualizaciones
        print(f"\n📊 Generando visualizaciones...")
        
        # Matriz de confusión
        self.plot_confusion_matrix(y_true, y_pred, class_names)
        
        # Curvas ROC y Precision-Recall (solo para clasificación binaria)
        if y_pred_proba is not None and len(np.unique(y_true)) == 2:
            auc_score = self.plot_roc_curve(y_true, y_pred_proba, model_name)
            ap_score = self.plot_precision_recall_curve(y_true, y_pred_proba, model_name)
            
            print(f"\n🎯 Métricas de Curvas:")
            print(f"   ROC AUC: {auc_score:.4f}")
            print(f"   Average Precision: {ap_score:.4f}")
        
        # Compilar reporte
        report = {
            'model_name': model_name,
            'basic_metrics': basic_metrics,
            'confusion_matrix_metrics': cm_metrics,
            'classification_report': classification_report(y_true, y_pred, 
                                                          target_names=class_names, 
                                                          output_dict=True)
        }
        
        return report
    
    def save_metrics_history(self, metrics: Dict[str, float], 
                            model_name: str, 
                            experiment_name: str = "default") -> None:
        """
        Guarda métricas en el historial.
        
        Args:
            metrics (Dict[str, float]): Métricas a guardar
            model_name (str): Nombre del modelo
            experiment_name (str): Nombre del experimento
        """
        
        entry = {
            'experiment': experiment_name,
            'model': model_name,
            'timestamp': pd.Timestamp.now(),
            **metrics
        }
        
        self.metrics_history.append(entry)
        print(f"💾 Métricas guardadas para {model_name} en experimento '{experiment_name}'")
    
    def get_metrics_history(self) -> pd.DataFrame:
        """
        Obtiene el historial de métricas como DataFrame.
        
        Returns:
            pd.DataFrame: Historial de métricas
        """
        
        if not self.metrics_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.metrics_history)
    
    def plot_metrics_evolution(self, metric: str = 'accuracy') -> None:
        """
        Plotea la evolución de una métrica a través del tiempo.
        
        Args:
            metric (str): Métrica a plotear
        """
        
        if not self.metrics_history:
            print("❌ No hay historial de métricas")
            return
        
        df = self.get_metrics_history()
        
        if metric not in df.columns:
            print(f"❌ Métrica '{metric}' no encontrada en el historial")
            return
        
        plt.figure(figsize=(12, 6))
        
        # Agrupar por modelo
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            plt.plot(model_data['timestamp'], model_data[metric], 
                    marker='o', linewidth=2, label=model)
        
        plt.xlabel('Tiempo')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'Evolución de {metric.replace("_", " ").title()} por Modelo')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Ejemplo de uso
    print("🔄 Probando módulo de métricas...")
    
    # Crear datos de ejemplo
    np.random.seed(42)
    n_samples = 1000
    
    y_true = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    y_pred_proba = np.random.random(n_samples)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Crear calculadora de métricas
    calc = MetricsCalculator()
    
    # Generar reporte
    report = calc.generate_detailed_report(
        y_true, y_pred, y_pred_proba, 
        class_names=['No Churn', 'Churn'],
        model_name="Modelo de Ejemplo"
    )
    
    print(f"\n✅ Ejemplo completado exitosamente")
