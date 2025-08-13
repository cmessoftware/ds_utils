"""
Clase para análisis exploratorio de datos (EDA) del proyecto de target.
"""

import warnings
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# Configuración global para visualizaciones
plt.style.use("default")
sns.set_palette("husl")


    
class EDA:
    """
    Clase para realizar análisis exploratorio de datos.
    """
    
    def __init__(self, df: pd.DataFrame, target_col: str = "target"):
        """
        Inicializa la clase EDA.
        
        Args:
            df (pd.DataFrame): Dataset a analizar
            target_col (str): Nombre de la columna objetivo
        """
        self.df = df.copy()
        self.target_col = target_col
        
    def basic_info(self, name: str = "Dataset") -> None:
        """
        Muestra información básica del dataset.
        
        Args:
            name (str): Nombre descriptivo del dataset
        """
        print(f"📊 INFORMACIÓN BÁSICA DE {name.upper()}")
        print("=" * 60)

        print(f"Dimensiones: {self.df.shape[0]:,} filas × {self.df.shape[1]} columnas")
        print(f"Memoria utilizada: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        print("\n🔍 Tipos de datos:")
        dtype_counts = self.df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"   - {dtype}: {count} columnas")

        print("\n⚠️ Valores faltantes:")
        missing = self.df.isnull().sum()
        total_missing = missing.sum()

        if total_missing > 0:
            missing_pct = (missing / len(self.df)) * 100
            missing_df = pd.DataFrame(
                {
                    "Columna": missing.index,
                    "Faltantes": missing.values,
                    "Porcentaje": missing_pct.values,
                }
            )
            missing_df = missing_df[missing_df["Faltantes"] > 0].sort_values(
                "Faltantes", ascending=False
            )
            print(missing_df.to_string(index=False))
        else:
            print("   ✅ No hay valores faltantes")

        print("\n📋 Resumen de columnas:")
        print(f"   - Numéricas: {len(self.df.select_dtypes(include=[np.number]).columns)}")
        print(f"   - Categóricas: {len(self.df.select_dtypes(include=['object']).columns)}")
        print(f"   - Booleanas: {len(self.df.select_dtypes(include=['bool']).columns)}")

    def analyze_target(self) -> None:
        """
        Analiza la variable objetivo.
        """
        print(f"🎯 ANÁLISIS DE VARIABLE OBJETIVO: {self.target_col}")
        print("=" * 60)

        if self.target_col not in self.df.columns:
            print(f"❌ La columna '{self.target_col}' no existe en el dataset")
            return

        # Contar valores
        value_counts = self.df[self.target_col].value_counts().sort_index()
        value_pcts = self.df[self.target_col].value_counts(normalize=True).sort_index() * 100

        print(f"Distribución de {self.target_col}:")
        for value, count, pct in zip(
            value_counts.index, value_counts.values, value_pcts.values
        ):
            print(f"   - {value}: {count:,} ({pct:.1f}%)")

        # Calcular desbalance
        if len(value_counts) == 2:
            minority_pct = min(value_pcts.values)
            if minority_pct < 30:
                print(f"\n⚠️ Dataset desbalanceado: clase minoritaria {minority_pct:.1f}%")
            else:
                print(
                    f"\n✅ Dataset balanceado: diferencia {abs(value_pcts.iloc[0] - value_pcts.iloc[1]):.1f}%"
                )

        # Visualización
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Gráfico de barras
        value_counts.plot(kind="bar", ax=axes[0], color=["lightblue", "salmon","red", "blue", "green", "black", "white", "orange", "purple", "cyan", "pink", "gray"])
        axes[0].set_title(f"Distribución de {self.target_col}")
        axes[0].set_xlabel(self.target_col)
        axes[0].set_ylabel("Frecuencia")
        axes[0].tick_params(axis="x", rotation=0)

        # Gráfico de torta
        value_pcts.plot(
            kind="pie", ax=axes[1], autopct="%1.1f%%", colors=["lightblue", "salmon","red", "blue", "green", "black", "white", "orange", "purple", "cyan", "pink", "gray"]
        )
        axes[1].set_title(f"Proporción de {self.target_col}")
        axes[1].set_ylabel("")

        plt.tight_layout()
        plt.show()

    def analyze_numerical_features(self) -> None:
        """
        Analiza las características numéricas del dataset.
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_col and self.target_col in numeric_cols:
            numeric_cols.remove(self.target_col)

        if not numeric_cols:
            print("❌ No hay columnas numéricas para analizar")
            return

        print("📈 ANÁLISIS DE CARACTERÍSTICAS NUMÉRICAS")
        print("=" * 60)
        print(f"Columnas numéricas encontradas: {len(numeric_cols)}")

        # Estadísticas descriptivas
        print("\n📊 Estadísticas descriptivas:")
        stats = self.df[numeric_cols].describe()
        print(stats.round(2))

        # Detectar outliers usando IQR
        print("\n🔍 Detección de outliers (método IQR):")
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            outlier_pct = len(outliers) / len(self.df) * 100

            print(f"   - {col}: {len(outliers)} outliers ({outlier_pct:.1f}%)")

        # Visualizaciones
        n_cols = min(len(numeric_cols), 4)
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

        if self.target_col and self.target_col in self.df.columns:
            # Boxplots por target
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1) if n_cols > 1 else [axes]

            for i, col in enumerate(numeric_cols):
                row, col_idx = divmod(i, n_cols)
                ax = axes[row][col_idx] if n_cols > 1 else axes[row]
                self.df.boxplot(column=col, by=self.target_col, ax=ax,color='lightgreen')
                ax.set_title(f"{col} por {self.target_col}")
                ax.set_xlabel(self.target_col)

            # Ocultar subplots vacíos
            for i in range(len(numeric_cols), n_rows * n_cols):
                row, col_idx = divmod(i, n_cols)
                axes[row][col_idx].set_visible(False)

            plt.tight_layout()
            plt.show()

        # Histogramas
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1) if n_cols > 1 else [axes]

        for i, col in enumerate(numeric_cols):
            row, col_idx = divmod(i, n_cols)
            ax = axes[row][col_idx] if n_cols > 1 else axes[row]

            self.df[col].hist(bins=30, ax=ax, alpha=0.7)
            ax.set_title(f"Distribución de {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Frecuencia")

        # Ocultar subplots vacíos
        for i in range(len(numeric_cols), n_rows * n_cols):
            row, col_idx = divmod(i, n_cols)
            axes[row][col_idx].set_visible(False)

        plt.tight_layout()
        plt.show()

    def analyze_categorical_features(self) -> None:
        """
        Analiza las características categóricas del dataset.
        """
        categorical_cols = self.df.select_dtypes(include=["object"]).columns.tolist()
        if self.target_col and self.target_col in categorical_cols:
            categorical_cols.remove(self.target_col)

        if not categorical_cols:
            print("❌ No hay columnas categóricas para analizar")
            return

        print("📊 ANÁLISIS DE CARACTERÍSTICAS CATEGÓRICAS")
        print("=" * 60)
        print(f"Columnas categóricas encontradas: {len(categorical_cols)}")

        # Análisis de cardinalidad
        print("\n🔢 Cardinalidad de variables categóricas:")
        for col in categorical_cols:
            unique_count = self.df[col].nunique()
            unique_pct = unique_count / len(self.df) * 100
            most_frequent = self.df[col].mode().iloc[0] if len(self.df[col].mode()) > 0 else "N/A"
            most_frequent_count = self.df[col].value_counts().iloc[0] if len(self.df) > 0 else 0
            most_frequent_pct = most_frequent_count / len(self.df) * 100

            print(f"   - {col}: {unique_count} valores únicos ({unique_pct:.1f}%)")
            print(
                f"     Más frecuente: '{most_frequent}' ({most_frequent_count}, {most_frequent_pct:.1f}%)"
            )

        # Visualizaciones
        n_cols = min(len(categorical_cols), 2)
        n_rows = (len(categorical_cols) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1) if n_cols > 1 else [axes]

        for i, col in enumerate(categorical_cols):
            row, col_idx = divmod(i, n_cols)
            ax = axes[row][col_idx] if n_cols > 1 else axes[row]

            # Limitar a top 10 categorías si hay muchas
            value_counts = self.df[col].value_counts()
            if len(value_counts) > 10:
                value_counts = value_counts.head(10)
                title = f"Top 10 categorías de {col}"
            else:
                title = f"Distribución de {col}"

            value_counts.plot(kind="bar", ax=ax)
            ax.set_title(title)
            ax.set_xlabel("")
            ax.set_ylabel("Frecuencia")
            ax.tick_params(axis="x", rotation=45)

        # Ocultar subplots vacíos
        for i in range(len(categorical_cols), n_rows * n_cols):
            row, col_idx = divmod(i, n_cols)
            axes[row][col_idx].set_visible(False)

        plt.tight_layout()
        plt.show()

        # Análisis bivariado con target si está disponible
        if self.target_col and self.target_col in self.df.columns:
            print(f"\n🎯 Relación con variable objetivo ({self.target_col}):")

            for col in categorical_cols[:5]:  # Limitar a 5 variables para no saturar
                print(f"\n   📊 {col} vs {self.target_col}:")

                # Tabla de contingencia
                crosstab = pd.crosstab(self.df[col], self.df[self.target_col], normalize="index") * 100
                print(crosstab.round(1))

    def correlation_analysis(self, threshold: float = 0.7) -> None:
        """
        Analiza correlaciones entre variables numéricas.
        
        Args:
            threshold (float): Umbral para detectar correlaciones altas
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) < 2:
            print(
                "❌ Se necesitan al menos 2 columnas numéricas para análisis de correlación"
            )
            return

        print("🔗 ANÁLISIS DE CORRELACIONES")
        print("=" * 60)

        # Calcular matriz de correlación
        correlation_matrix = self.df[numeric_cols].corr()

        # Detectar correlaciones altas
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if pd.notna(corr_val) and abs(float(corr_val)) >= threshold:
                    high_corr_pairs.append(
                        (
                            correlation_matrix.columns[i],
                            correlation_matrix.columns[j],
                            correlation_matrix.iloc[i, j],
                        )
                    )

        if high_corr_pairs:
            print(f"⚠️ Correlaciones altas (|r| >= {threshold}):")
            for var1, var2, corr in high_corr_pairs:
                print(f"   - {var1} ↔ {var2}: {corr:.3f}")
        else:
            print(f"✅ No se encontraron correlaciones altas (|r| >= {threshold})")

        # Correlaciones con variable objetivo
        if self.target_col and self.target_col in numeric_cols:
            target_corrs = (
                correlation_matrix[self.target_col]
                .drop(self.target_col)
                .abs()
                .sort_values(ascending=False)
            )
            print(f"\n🎯 Correlaciones con {self.target_col} (ordenadas por magnitud):")
            for var, corr in target_corrs.items():
                print(f"   - {var}: {corr:.3f}")

        # Visualización del mapa de calor
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(
            correlation_matrix,
            mask=mask,
            annot=True,
            cmap="coolwarm",
            center=0,
            square=True,
            fmt=".2f",
            cbar_kws={"shrink": 0.8},
        )
        plt.title("Matriz de Correlaciones")
        plt.tight_layout()
        plt.show()

        # Mapa de calor de la correlación entre la variable objetivo y las otras características
        if self.target_col and self.target_col in numeric_cols:
            target_correlation = correlation_matrix[self.target_col].sort_values(ascending=False)

            plt.figure(figsize=(8, 10))
            sns.heatmap(
                target_correlation.to_frame(), annot=True, cmap="coolwarm", fmt=".2f"
            )
            plt.title(f"Correlation with {self.target_col}")
            plt.show()
        else:
            print(
                f"⚠️ No se puede mostrar correlaciones con variable objetivo: '{self.target_col}' no está en las columnas numéricas o no se especificó."
            )

    def analyze_target_patterns(self, variable: str, title: str) -> pd.Series:
        """
        Analiza los patrones de target para una variable específica.
        
        Args:
            variable (str): Nombre de la variable a analizar
            title (str): Título para los gráficos
            
        Returns:
            pd.Series: Tasas de target por categoría
        """
        # Crear tabla de contingencia
        crosstab = pd.crosstab(self.df[variable], self.df[self.target_col], normalize="index") * 100
        crosstab_counts = pd.crosstab(self.df[variable], self.df[self.target_col])

        # Crear figura con subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Gráfico de barras con porcentajes
        crosstab.plot(kind="bar", ax=ax1, color=["lightblue", "salmon"])
        ax1.set_title(f"{title} - Tasa de target (%)")
        ax1.set_xlabel(variable)
        ax1.set_ylabel("Porcentaje")
        ax1.legend(["No target", "target"])
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right")

        # Agregar etiquetas de porcentaje
        for container in ax1.containers:
            ax1.bar_label(container, fmt="%.1f%%")

        # Gráfico de barras apiladas con conteos
        crosstab_counts.plot(
            kind="bar", stacked=True, ax=ax2, color=["lightblue", "salmon"]
        )
        ax2.set_title(f"{title} - Distribución Total")
        ax2.set_xlabel(variable)
        ax2.set_ylabel("Cantidad de Clientes")
        ax2.legend(["No target", "target"])
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")

        plt.tight_layout()
        plt.show()

        # Mostrar estadísticas detalladas
        print(f"\n📊 ANÁLISIS DETALLADO: {title.upper()}")
        print("=" * 60)

        # Tabla de contingencia con conteos
        print(f"\n📋 Conteos por {variable}:")
        print(crosstab_counts)

        # Tabla de contingencia con porcentajes
        print(f"\n📈 Tasa de target por {variable} (%):")
        target_rates = crosstab.iloc[:, 1].sort_values(ascending=False)
        for category in target_rates.index:
            total = crosstab_counts.loc[category].sum()
            target_count = crosstab_counts.loc[category].iloc[1]
            target_rate = target_rates[category]
            print(f"   {category}: {target_rate:.1f}% ({target_count}/{total} clientes)")

        # Identificar patrones clave
        highest_target = target_rates.index[0]
        lowest_target = target_rates.index[-1]

        print(f"\n🔍 PATRONES IDENTIFICADOS:")
        print(
            f"   ⚠️  MAYOR RIESGO: {highest_target} ({target_rates[highest_target]:.1f}% target)"
        )
        print(
            f"   ✅ MENOR RIESGO: {lowest_target} ({target_rates[lowest_target]:.1f}% target)"
        )
        print(
            f"   📊 DIFERENCIA: {target_rates[highest_target] - target_rates[lowest_target]:.1f} puntos porcentuales"
        )

        return target_rates

    def show_correlation_respect_to_feature(self, feature: str = 'target_Yes') -> None:
        """
        Muestra correlación con respecto a una característica específica.
        
        Args:
            feature (str): Nombre de la característica de referencia
        """
        # Verifica que la característica existe
        if feature not in self.df.columns:
            raise ValueError(f"⚠️ La columna '{feature}' no está presente en el dataset.")
        
        # Seleccionar columnas numéricas + booleanas
        numeric_df = self.df.select_dtypes(include=['int64', 'float64', 'bool']).astype(float)
        
        # Calcular matriz de correlación
        correlation_matrix = numeric_df.corr()
        
        # Extraer correlación con respecto a la característica (sin autocorrelación)
        target_corr = correlation_matrix[feature].drop(feature).sort_values()
        
        # Mostrar heatmap vertical
        plt.figure(figsize=(6, len(target_corr) * 0.5))
        sns.heatmap(target_corr.to_frame(), annot=True, cmap='coolwarm', center=0, cbar=True)
        plt.title(f'📊 Correlación con {feature}', fontsize=14)
        plt.xlabel('Correlación')
        plt.ylabel('Variable')
        plt.tight_layout()
        plt.show()

    def generate_full_report(self) -> None:
        """
        Genera un reporte completo de EDA.
        """
        print("🔍 GENERANDO REPORTE COMPLETO DE EDA")
        print("=" * 80)

        # Información básica
        self.basic_info("Dataset Principal")
        print("\n" + "=" * 80 + "\n")

        # Análisis de variable objetivo
        self.analyze_target()
        print("\n" + "=" * 80 + "\n")

        # Análisis de características numéricas
        self.analyze_numerical_features()
        print("\n" + "=" * 80 + "\n")

        # Análisis de características categóricas
        self.analyze_categorical_features()
        print("\n" + "=" * 80 + "\n")

        # Análisis de correlaciones
        self.correlation_analysis()
        print("\n" + "=" * 80 + "\n")

        print("✅ Reporte de EDA completado")


def generate_full_report(df: pd.DataFrame, target_col: str = "target", 
           name: str = "Dataset", correlation_threshold: float = 0.7) -> None:
    """
    Legacy function for backward compatibility. Runs complete EDA analysis.
    
    Args:
        df (pd.DataFrame): Dataset to analyze
        target_col (str): Name of target column
        name (str): Descriptive name for the dataset
        correlation_threshold (float): Threshold for high correlations
    """
    warnings.warn(
        "generate_full_report() is deprecated. Use EDA class instead: EDA(df, target_col).generate_full_report()",
        DeprecationWarning,
        stacklevel=2
    )
    
    eda = EDA(df, target_col)
    eda.basic_info(name)
    print("\n" + "=" * 80 + "\n")
    eda.analyze_target()
    print("\n" + "=" * 80 + "\n")
    eda.analyze_numerical_features()
    print("\n" + "=" * 80 + "\n")
    eda.analyze_categorical_features()
    print("\n" + "=" * 80 + "\n")
    eda.correlation_analysis(correlation_threshold)
    print("\n" + "=" * 80 + "\n")
    print("✅ Legacy EDA analysis completed")


def basic_info(df: pd.DataFrame, name: str = "Dataset") -> None:
    """
    Legacy function for basic information.
    
    Args:
        df (pd.DataFrame): Dataset to analyze
        name (str): Descriptive name for the dataset
    """
    warnings.warn(
        "basic_info() is deprecated. Use EDA(df).basic_info(name) instead.",
        DeprecationWarning,
        stacklevel=2
    )

    eda = EDA(df)
    eda.basic_info(name)


def analyze_target(df: pd.DataFrame, target_col: str = "target") -> None:
    """
    Legacy function for target analysis.
    
    Args:
        df (pd.DataFrame): Dataset to analyze
        target_col (str): Name of target column
    """
    warnings.warn(
        "analyze_target() is deprecated. Use EDA(df, target_col).analyze_target()",
        DeprecationWarning,
        stacklevel=2
    )
    
    eda = EDA(df, target_col)
    eda.analyze_target()

def analyze_numerical_features(df: pd.DataFrame) -> None:
    """
    Legacy function for numerical features analysis.

    Args:
        df (pd.DataFrame): Dataset to analyze
    """
    warnings.warn(
        "analyze_numerical_features() is deprecated. Use EDA(df).analyze_numerical_features()",
        DeprecationWarning,
        stacklevel=2
    )

    eda = EDA(df)
    eda.analyze_numerical_features()
    
def analyze_categorical_features(df: pd.DataFrame) -> None:
    """
    Legacy function for categorical features analysis.

    Args:
        df (pd.DataFrame): Dataset to analyze
    """
    warnings.warn(
        "analyze_categorical_features() is deprecated. Use EDA(df).analyze_categorical_features()",
        DeprecationWarning,
        stacklevel=2
    )

    eda = EDA(df)
    eda.analyze_categorical_features()

def correlation_analysis(df: pd.DataFrame, threshold: float = 0.7) -> None:
    """
    Legacy function for correlation analysis.
    
    Args:
        df (pd.DataFrame): Dataset to analyze
        threshold (float): Threshold for high correlations
    """
    warnings.warn(
        "correlation_analysis() is deprecated. Use EDA(df).correlation_analysis(threshold)",
        DeprecationWarning,
        stacklevel=2
    )
    
    eda = EDA(df)
    eda.correlation_analysis(threshold)
    