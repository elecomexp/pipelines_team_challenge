import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import math

import warnings

from scipy.stats import chi2_contingency, f_oneway, mannwhitneyu, pearsonr
from sklearn.feature_selection import f_regression



######################################
#   COMUNES A TODOS LOS DATA FRAME   #
######################################

def ALL_describe_features(df:pd.DataFrame,umbral_categoria=10, umbral_continua=30.0) -> pd.DataFrame:
 # Verificar que el argumento es un DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Se esparaba un Data Frame como argumento de entrada")
        
    if not isinstance(umbral_categoria, int) or not isinstance(umbral_continua, (int, float)):
        raise TypeError("Los umbrales deben de ser números enteros o exactos")
    

    cardinality = df.nunique()
    cardinality_percentage = (cardinality / len(df)) * 100

    tipo_sugerido = []
    for col in df.columns:
        unique_count = cardinality[col]
        if unique_count == 2:
            tipo_sugerido.append("Binaria")
          
        elif unique_count < umbral_categoria:
                tipo_sugerido.append("Categórica")
              
        else:
            if cardinality_percentage[col] >= umbral_continua:
                    tipo_sugerido.append("Numérica Continua")
                    
            else:
                    tipo_sugerido.append("Numérica Discreta")
                
    
    lista_missings = []
    lista_no_missings = []
    for col in df.columns:
        if df[col].isnull().mean() == 0:
             lista_missings.append(0)
             lista_no_missings.append(df.shape[0])
        else:
             missings = df[col].isna().value_counts()[True]
             no_missings = df[col].isna().value_counts()[False]
             lista_missings.append(missings)
             lista_no_missings.append(no_missings)
     

    # Diccionario con las estadísticas
    data = {
        'COL_N': df.columns,
        'DATA_TYPE': df.dtypes.values,
        'NO MISSING': lista_no_missings,
        'MISSING' : lista_missings,
        'MISSING (%)': df.isnull().mean().values * 100,
        'UNIQUE_VALUES': df.nunique().values,
        'CARDIN (%)': (df.nunique() / len(df)).values * 100,
        'DATA_CLASS': tipo_sugerido
        }
    
    # Crear el DataFrame final y establecer la columna 'COL_N' como índice
    df_out = round(pd.DataFrame(data),2)
    #df_out.set_index('COL_N', inplace=True)
    
    return df_out

def ALL_lista_features(df, umbral_categoria=10, umbral_continua=30):

 # Verificar que el argumento es un DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Se esparaba un Data Frame como argumento de entrada")
        
    if not isinstance(umbral_categoria, int) or not isinstance(umbral_continua, (int, float)):
        raise TypeError("Los umbrales deben de ser números enteros o exactos")
    

    cardinality = df.nunique()
    cardinality_percentage = (cardinality / len(df)) * 100

    lista_num = []
    lista_cat = []

    for col in df.columns:
        unique_count = cardinality[col]
        if unique_count == 2:
            lista_cat.append(col)

        elif unique_count < umbral_categoria:
                lista_cat.append(col)
        else:
            if (cardinality_percentage[col] >= umbral_continua and pd.api.types.is_numeric_dtype(df[col])) or (unique_count > umbral_categoria and pd.api.types.is_numeric_dtype(df[col])):                
                    lista_num.append(col)
            else:
                    lista_cat.append(col)

    print(f"Num: {lista_num}, Cat: {lista_cat}")

    return None


########################
#   PARA REGRESIONES   # 
########################

## FEATURES NÚMERICAS ##

def REGRE_FN_get_features_num_regression(df:pd.DataFrame, target_col:str, umbral_corr:float, pvalue:float=None, umbral_card_target=10.0, umbral_categorias_feature=10, pearson_results=False) -> list:
    """
    Obtiene las columnas numéricas de un DataFrame cuya correlación con la columna objetivo 
    supera un umbral especificado. Además, permite filtrar las columnas en función 
    de la significancia estadística de la correlación, mediante un valor-p opcional.

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame que contiene los datos a analizar.

    target_col : str
        Nombre de la columna objetivo que se desea predecir; debe ser 
        una variable numérica continua o discreta con alta cardinalidad.

    umbral_corr : float
        Umbral de correlación absoluta para considerar una relación 
        significativa entre las columnas (debe estar comprendido entre 0 y 1).

    pvalue : float (opcional)
        Valor-p que determina el nivel de significancia para 
        filtrar las columnas. Si se proporciona, solo se incluirán 
        las columnas cuya correlación supere el umbral `umbral_corr` y cuyo 
        valor-p sea menor que `pvalue`, es decir, las que tengan una 
        significancia estadística mayor o igual a 1-p_value.
        Debe estar comprendido entre 0 y 1.

    umbral_card_target : float (opcional)
        Umbral para definir una alta cardinalidad en una variable numérica.
        Si la cardinalidad porcentual del target_col es superior o igual a este umbral, entonces se 
        considera que la columna tiene una alta cardinalidad. En otro caso, tiene una baja cardinalidad.
        
    umbral_categoria : int (opcional)
        Umbral para considerar una variable como categórica en función de su cardinalidad.
        Su valor por defecto es 10.
        
    pearson_results : bool (opcional)
        Si es `True`, imprime los resultados del test de Pearson para cada columna que
        cumpla los criterios de correlación y significancia. Los resultados incluyen el
        nombre de la columna, el valor de correlación y el p-valor correspondiente.

    Retorna:
    --------
    lista_num : list
        Lista de nombres de columnas numéricas que cumplen con los criterios establecidos.
        Si no hay columnas que cumplan los requisitos, se devuelve una lista vacía.
        Si algún argumento no es válido, se devuelve None.

    Excepciones:
    -----------
    La función imprime mensajes de error en los siguientes casos:
    - Si `df` no es un DataFrame.
    - Si `target_col` no es una columna del DataFrame.
    - Si `target_col` no es una variable numérica continua o es discreta con baja cardinalidad.
    - Si `umbral_corr` no es un número entre 0 y 1.
    - Si `pvalue` no es None y no es un número entre 0 y 1.

    En cualquiera de estos casos, la función retorna `None`.
    """
    # Comprobaciones iniciales de los argumentos
    if not _is_dataframe(df):
        return None
    
    if target_col not in df.columns:
        print(f"Error: {target_col} no es una columna del DataFrame.")
        return None
    
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print(f"Error: {target_col} no es una columna numérica.")
        return None
    
    if not isinstance(umbral_card_target, (int, float)):
        print(f"Error: {umbral_card_target} no es un valor válido para 'umbral_card'. Debe ser un float.")
        return None

    # Cardinalidad de la columna "target"
    cardinality_percentage = (df[target_col].nunique() / len(df)) * 100 
    if cardinality_percentage < umbral_card_target:
        print(f"Error: {target_col} tiene una cardinalidad inferior a {umbral_card_target}.")
        return None

    if not isinstance(umbral_corr, (int, float)) or umbral_corr < 0 or umbral_corr > 1:
        print(f"Error: {umbral_corr} no es un valor válido para 'umbral_corr'. Debe estar entre 0 y 1.")
        return None
    
    if pvalue is not None and (not isinstance(pvalue, (int, float)) or pvalue < 0 or pvalue > 1):
        print(f"Error: {pvalue} no es un valor válido para 'pvalue'. Debe estar entre 0 y 1.")
        return None

    # Creación de la lista de columnas numéricas que cumplen con los criterios establecidos
    lista_num = []
    corr = []
    p_val = []
    direccion = []
    for columna in df.columns:
        if pd.api.types.is_numeric_dtype(df[columna]) and columna != target_col and df.nunique()[columna] > umbral_categorias_feature:
            resultado_test = pearsonr(df[columna], df[target_col])
            correlacion = resultado_test[0]
            p_valor = resultado_test[1]
            
            if np.isnan(correlacion):
                print(f"La columna {columna} no ha podido ser calculada. Revisa sus valores en busca de nulos.")
                print("")

            else:  
                if abs(correlacion) > umbral_corr:
                    # Confianza del 1-p_valor
                    if pvalue is None or p_valor < pvalue:  
                        lista_num.append(columna)
                        corr.append(abs(correlacion))
                        direccion.append("-" if correlacion < 0 else "+")
                        p_val.append(p_valor)

    if pearson_results:
        print("## Test de Person para correlación entre variables y su significancia ##")
    
        pearson_df = pd.DataFrame({'Feature': lista_num, 'Corr': corr, "Direction":direccion, 'p-value': p_val}).sort_values(by="Corr", ascending=False)
        
        print(pearson_df)
        lista_num = pearson_df["Feature"]
    
    print("")          
    
    return lista_num

def REGRE_FN_plot_features_num_regression(df:pd.DataFrame, target_col='', columns=[], umbral_corr=0.0, pvalue=None, umbral_card=10.0) -> list:
    """
    Visualiza las relaciones entre una columna objetivo y las columnas numéricas del DataFrame que cumplen con los criterios 
    de correlación y significancia especificados. Utiliza pairplots de Seaborn para mostrar las relaciones entre las 
    columnas seleccionadas.

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame que contiene los datos a analizar.

    target_col : str, opcional
        Nombre de la columna objetivo que se desea predecir; debe ser una variable numérica continua o discreta 
        con alta cardinalidad.

    columns : list, opcional
        Lista de nombres de columnas numéricas a considerar para la visualización. Si se proporciona, solo se 
        visualizarán las columnas en esta lista que también cumplen con los criterios establecidos. Si se omite, 
        se utilizarán todas las columnas numéricas que cumplen con los criterios.

    umbral_corr : float, opcional
        Umbral de correlación absoluta para considerar una relación significativa entre las columnas (debe estar 
        comprendido entre 0 y 1). Solo se visualizarán las columnas cuya correlación con `target_col` sea mayor 
        que este umbral.

    pvalue : float, opcional
        Valor-p que determina el nivel de significancia para filtrar las columnas. Si se proporciona, solo se 
        incluirán las columnas cuya correlación supere el umbral y cuyo valor-p sea menor que `pvalue`, es decir, 
        las que tengan una significancia estadística mayor o igual a 1 - pvalue. Debe estar comprendido entre 0 y 1.

    umbral_card : float, opcional
        Umbral para definir una alta cardinalidad en una variable numérica. Si la cardinalidad porcentual del 
        `target_col` es superior o igual a este umbral, entonces se considera que la columna tiene una alta 
        cardinalidad. En otro caso, tiene una baja cardinalidad.

    Retorna:
    --------
    list
        Lista de nombres de columnas numéricas que cumplen con los criterios establecidos y que se han utilizado 
        para crear los pairplots. Si no hay columnas que cumplan los requisitos, se devuelve una lista vacía.

    Excepciones:
    -----------
    La función imprime mensajes de error en los siguientes casos:
    - Si `target_col` no está en el DataFrame.
    - Si ninguna columna cumple con los criterios de correlación y significancia.
    - Si ocurre algún problema al generar los pairplots.

    Ejemplo:
    --------
    >>> plot_features_num_regression(df, 'median_house_value', umbral_corr=0.1, pvalue=0.05, umbral_card=12.5)
    """

    # Obtener lista de features relevantes a través de get_features_num_regression 
    lista = get_features_num_regression(df, target_col, umbral_corr, pvalue, umbral_card)
    
    # Gestión errores heredados de get_features_num_regression()
    if lista is None:
        return None
    elif not lista:
        print('Error: Ninguna columna cumple con los criterios de correlación y significancia.')
        return None
    
    # Si no se han especificado columnas, usar las obtenidas de get_features_num_regression
    if not columns:
        numeric_columns = lista
    else:
        # Filtrar las que cumplen con la significancia estadística
        numeric_columns = [col for col in columns if col in lista]
    
    # Si ninguna columa cumple con los criterios de correlación y confianza, retorna None
    if not numeric_columns:
        print("Error: Ninguna columna de 'columns' cumple con el criterio de significancia.")
        return None
      
    # Dividir en grupos de 5 para los pairplots (1 columna objetivo + 4 columnas adicionales)
    for i in range(0, len(numeric_columns), 4):
        subset_cols = numeric_columns[i:i + 4]
        # Asegurarse que target_col siempre esté en cada subset
        if target_col not in subset_cols:
            subset_cols.insert(0, target_col)  
        sns.pairplot(df[subset_cols])
        plt.show()

    return numeric_columns

def REGRE_FN_BI_FeaNum_hist_scatter_plot_with_regression(df, target_col, umbral_corr=0.0, pvalue= None, umbral_card_target=10.0, umbral_categorias_feature=10, pearson_results=True, f_regression=True):

    feature_list = get_features_num_regression(df, target_col, umbral_corr, pvalue, umbral_card_target, umbral_categorias_feature, pearson_results)

    if f_regression:
        f_regression_analysis(df,target_col,umbral_categorias_feature,umbral_card_target)

    for columna in feature_list:
        
        sns.jointplot(x= target_col,
                    y=columna,
                    data=df,kind='reg', 
                    color='blue',
                    scatter_kws={'color': 'red'})
        plt.title(f"{target_col} vs {columna}")
        plt.tight_layout()

    return list(feature_list)

def REGRE_FN_f_regression_analysis(df, target_col, umbral_categorias_feature=10, umbral_card_target=10.0):

    # Cardinalidad de la columna "target"

    print("## Test de Fuerza entre variables y su significancia ##")

    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print(f"Error: {target_col} no es una columna numérica.")
        return None

    cardinality_percentage = (df[target_col].nunique() / len(df)) * 100 
    if cardinality_percentage < umbral_card_target:
        print(f"Error: {target_col} tiene una cardinalidad inferior a {umbral_card_target}.")
        return None
    
    
    list_not_num = []
    list_low_nunique = []

    # Identificar columnas no numéricas para eliminarlas
    for columna in df.columns:
        if not pd.api.types.is_numeric_dtype(df[columna]) :
            list_not_num.append(columna)
        elif pd.api.types.is_numeric_dtype(df[columna]) and df[columna].nunique() < umbral_categorias_feature:
            list_low_nunique.append(columna)
        elif df[columna].isna().any():   
            print(f"Error: {columna} tiene nulos o missings. No es posible usarla para el cálculo. Limpia la columna y prueba de nuevo.")
            return None
    
    list_to_drop = list_not_num + list_low_nunique
    
        
    # Convertir a DataFrame para facilitar la visualización
    df_X = df.drop(columns=[target_col] + list_to_drop)
    df_y = df[target_col] 

    # Calcular los valores F y p usando f_regression
    F_values, p_values = f_regression(df_X, df_y)

    # Crear un DataFrame para visualizar los resultados
    results = pd.DataFrame({'Feature': df_X.columns, 'F-value': F_values, 'p-value': p_values})

    # Ordenar por p-value
    results_sorted = results.sort_values(by='p-value')

    if list_not_num != []:
        print(f"Las siguientes columnas se han quitado por NO ser numéricas: {list_not_num}")
    if list_low_nunique != []:    
        print(f"Las siguientes columnas númericas se han quitado por tener menos de {umbral_categorias_feature+1} valores únicos y considerarse categóricas: {list_low_nunique}")
    print("")
    print(results_sorted)

    return None

## FEATURES CATEGÓRICAS ##

def REGRE_FC_get_features_cat_regression(df:pd.DataFrame, target_col:str, pvalue=0.05, umbral_categoria=10, umbral_card=10.0, mann_results=False, anova_results=False) -> list:
    """
    La función devuelve una lista con las columnas categóricas del dataframe cuyo test de relación 
    con la columna designada por 'target_col' supera el umbral de confianza estadística definido por 'pvalue'.
    
    La función realiza una Prueba U de Mann-Whitney si la variable categórica es binaria,
    o una prueba ANOVA (análisis de varianza) si la variable categórica tiene más de dos niveles.

    La función también realiza varias comprobaciones previas para asegurar que los argumentos de entrada son adecuados. 
    Si alguna condición no se cumple, la función retorna 'None' y muestra un mensaje explicativo.

    Parámetros
    ----------
    df : pd.DataFrame
        Dataframe que contiene los datos a analizar.

    target_col : str
        Nombre de la columna objetivo que se desea predecir; debe ser una variable numérica continua 
        o discreta con alta cardinalidad.

    pvalue : float (opcional)
        Umbral de significancia estadística para los tests de relación. Su valor por defecto es 0.05.

    umbral_categoria : int (opcional)
        Umbral para considerar una variable como categórica en función de su cardinalidad.
        Su valor por defecto es 10.

    umbral_card : float (opcional)
        Porcentaje mínimo de valores únicos en relación al tamaño del dataframe por encima del cual 
        la variable numérica objetivo (target) se considera de alta cardinalidad. Su valor por defecto es 10.0.

    Retorna:
    lista_categoricas : list
        Lista de nombres de columnas categóricas que cumplen con los criterios establecidos.
        Si no hay columnas que cumplan los requisitos, se devuelve una lista vacía.
        Si las condiciones de entrada no se cumplen, se devuelve None.
    """

    # Comprobaciones
    if not _is_dataframe(df):
        return None
    
    if target_col not in df.columns:
        print("Error: La columna objetivo no existe en el DataFrame.")
        return None

    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print("Error: La columna objetivo debe ser de tipo numérico.")
        return None

    if not (0 <= pvalue <= 1):
        print("Error: El valor de pvalue debe estar entre 0 y 1.")
        return None

    # Comprobar si el target tiene alta cardinalidad
    cardinalidad_percent = (df[target_col].nunique() / df.shape[0]) * 100
    if cardinalidad_percent < umbral_card:
        print(f"Error: La columna objetivo no cumple con el umbral de alta cardinalidad ({umbral_card}%).")
        return None


    # Inicialización de listas
    lista_categoricas_anova = []
    lista_categoricas_mann = []
    lista_u_stat = []
    lista_p_val_mann = []
    lista_f_val = []
    lista_p_val_anova = []

    # Recorrer las columnas del DataFrame
    for col in df.columns:
        # Comprobar si la columna es categórica
        if pd.api.types.is_categorical_dtype(df[col]) or df[col].nunique() <= umbral_categoria:
            if df[col].nunique() == 1:
                print(f"IMPORTANTE: La columna `{col}` tiene un solo valor. Considera eliminarla.")
                print("")
            elif df[col].nunique() == 2:  # Binaria
                a = df.loc[df[col] == df[col].unique()[0], target_col]
                b = df.loc[df[col] == df[col].unique()[1], target_col]
                u_stat, p_val_mann = mannwhitneyu(a, b)
                if p_val_mann <= pvalue:
                    lista_categoricas_mann.append(col)
                    lista_u_stat.append(u_stat)
                    lista_p_val_mann.append(p_val_mann)
            else:  # Más de dos categorías
                grupos = [df[df[col] == nivel][target_col] for nivel in df[col].unique()]

                # Filtrar grupos vacíos
                grupos_validos = [grupo for grupo in grupos if len(grupo) > 1]

                if len(grupos_validos) >= 2:  # Asegurarse de que hay al menos dos grupos válidos
                    f_val, p_val_anova = f_oneway(*grupos_validos)
                    if p_val_anova <= pvalue:
                        lista_categoricas_anova.append(col)
                        lista_f_val.append(f_val)
                        lista_p_val_anova.append(p_val_anova)

    # Resultados Mann-Whitney U
    if mann_results and lista_categoricas_mann:
        print("## Test de Mann-Whitney U para medir diferencias entre los rangos de los grupos y su significancia ##")
        print("")
        df_mannwhitneyu_results = pd.DataFrame({
            "Feature": lista_categoricas_mann,
            "U Stat": lista_u_stat,
            "p_value": lista_p_val_mann
        })
        print(df_mannwhitneyu_results)

    # Resultados ANOVA
    if anova_results and lista_categoricas_anova:
        print("")
        print("## Test ANOVA para medir diferencias entre las medias de los grupos y su significancia ##")
        print("")
        df_anova_results = pd.DataFrame({
            "Feature": lista_categoricas_anova,
            "F Val": lista_f_val,
            "p_value": lista_p_val_anova
        })
        print(df_anova_results)

    return lista_categoricas_mann + lista_categoricas_anova

def REGRE_FC_plot_features_cat_regression(df:pd.DataFrame, target_col= "", columns=[], pvalue=0.05, with_individual_plot=False, umbral_categoria=10, umbral_card=10.0) -> list:
    """
    La función recibe un DataFrame y realiza un análisis de las columnas categóricas en relación con una columna objetivo numérica.
    Pinta histogramas agrupados de la variable objetivo por cada columna categórica seleccionada si su test de relación es estadísticamente significativo.
    
    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame que contiene los datos a analizar.
    
    target_col : str, opcional
        Nombre de la columna objetivo numérica (por defecto es "").
    
    columns : list, opcional
        Lista de nombres de las columnas categóricas a analizar (por defecto es lista vacía).
    
    pvalue : float, opcional
        Nivel de significancia estadística para los tests de relación (por defecto es 0.05).
    
    with_individual_plot : bool, opcional
        Si es True, genera un histograma separado para cada variable categórica. Si es False, los agrupa (por defecto es False).
    
    umbral_categoria : int, opcional
        Umbral de cardinalidad para considerar una columna como categórica (por defecto es 10).
    
    umbral_card : float, opcional
        Porcentaje mínimo de valores únicos en relación al tamaño del dataframe por encima del cual 
        la variable numérica objetivo (target) se considera de alta cardinalidad. Su valor por defecto es 10.0.    
    
    Retorna:
    -------
    list
        Lista de nombres de columnas categóricas que cumplen con los criterios de significancia estadística.
        Si no se cumplen las condiciones, retorna None o una lista vacía.
    """
    # Obtener las columnas categóricas con relación significativa usando get_features_cat_regression
    lista = get_features_cat_regression(df, target_col, pvalue, umbral_categoria, umbral_card)

    if lista is None:
        return None
    elif not lista:
        print('Error: Ninguna columna cumple con los criterios de correlación y significancia.')
        return None

    # Si 'columns' está vacía, tomar todas las variables categóricas que pasaron el test
    if not columns:
        columns = lista
    else:
        # Filtrar las que cumplen con la significancia estadística
        columns = [col for col in columns if col in lista]
    
    # Si ninguna columa cumple con los criterios de significancia, retorna None
    if not columns:
        print("Error: Ninguna columna de 'columns' cumple con el criterio de significancia.")
        return None

     # Plotear gráficos individuales
    if with_individual_plot:
        for col in columns:
            # Histograma
            sns.histplot(data=df, x=target_col, hue=col, kde=True, multiple="layer")
            plt.title(f"Histograma de {target_col} con {col}")
            plt.show()

            # Boxplot
            sns.boxplot(data=df, x=target_col, hue=col)
            plt.title(f"Boxplot de {target_col} con {col}")
            plt.show()
    else:
        # Plotear con subplots
        columnas_por_fila = 2  # Dos gráficos por fila (histogram y boxplot)
        filas_ploteo = math.ceil(len(columns) * 2 / columnas_por_fila)  # Ajustar filas para dos gráficos por columna

        fig, axes = plt.subplots(filas_ploteo, columnas_por_fila, figsize=(20, 5 * filas_ploteo), constrained_layout=True)

        # Aplanar ejes para facilitar la iteración
        axes = axes.flatten()

        for i, col in enumerate(columns):
            # Histograma
            sns.histplot(data=df, x=target_col, hue=col, ax=axes[i * 2], kde=True, multiple="layer")
            axes[i * 2].set_title(f"Histograma de {target_col} con {col}")

            # Boxplot
            sns.boxplot(data=df, x=target_col, hue=col, ax=axes[i * 2 + 1])
            axes[i * 2 + 1].set_title(f"Boxplot de {target_col} con {col}")

        # Ocultar ejes vacíos si hay menos gráficos que espacios de subplots
        for j in range(i * 2 + 2, len(axes)):
            axes[j].axis('off')

        plt.show()

    return columns

def REGRE_FC_categoricas_hue(df, target_col, columnas_categoricas):
    num_columnas = len(columnas_categoricas)
    num_filas = (num_columnas // 2) + (num_columnas % 2)

    fig, axes = plt.subplots(num_filas, 2, figsize=(15, 5 * num_filas))
    axes = axes.flatten()

    for i, col in enumerate(columnas_categoricas):
        ax = axes[i]

        # Crear el countplot
        sns.countplot(data=df, x=target_col, ax=ax, hue=col)

        # Mostrar los valores encima de las barras
        for p in ax.patches:
            height = p.get_height()  # Obtener la altura de la barra
            ax.annotate(f'{int(height)}', 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha='center', va='bottom', 
                        xytext=(0, 0), textcoords='offset points')  # Ajustar el desplazamiento

        ax.set_title(f'Distribución de {col}')
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45)

   

    plt.tight_layout()
    plt.show()



############################
#   PARA CLASIFICACIONES   #
############################

## NUMERICAS ##

def CLASI_UNI_FeaNum_hist_box_plot(df, columns, whisker_width=1.5, bins = None):
    num_cols = len(columns)
    if num_cols:
        
        fig, axes = plt.subplots(num_cols, 2, figsize=(12, 5 * num_cols))
        print(axes.shape)

        for i, column in enumerate(columns):
            if df[column].dtype in ['int64', 'float64']:
                # Histograma y KDE
                sns.histplot(df[column], kde=True, ax=axes[i,0] if num_cols > 1 else axes[0], bins= "auto" if not bins else bins)
                if num_cols > 1:
                    axes[i,0].set_title(f'Histograma y KDE de {column}')
                else:
                    axes[0].set_title(f'Histograma y KDE de {column}')

                # Boxplot
                sns.boxplot(x=df[column], ax=axes[i,1] if num_cols > 1 else axes[1], whis=whisker_width)
                if num_cols > 1:
                    axes[i,1].set_title(f'Boxplot de {column}')
                else:
                    axes[1].set_title(f'Boxplot de {column}')

        plt.tight_layout()
        plt.show()

def CLASI_BI_FeaNum_hist_kde_plot(df, features_num_iniciales, target, num_graf_per_line = 2):
    num_features = len(features_num_iniciales)
    num_cols = num_graf_per_line  
    num_rows = (num_features // num_cols) + (num_features % num_cols > 0)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    axes = axes.flatten()  

    for i, col in enumerate(features_num_iniciales):
        sns.histplot(data=df, x=col, hue=target, ax=axes[i], kde=True)  
        axes[i].set_title(f'Histograma Bivariante de {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frecuencia')

    
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

## CATEGORICAS ##

def CLASI_UNI_FeaCat_bar_plot(df, columnas_categoricas, relativa=False, mostrar_valores=False):
    num_columnas = len(columnas_categoricas)
    num_filas = (num_columnas // 2) + (num_columnas % 2)
    fig, axes = plt.subplots(num_filas, 2, figsize=(15, 5 * num_filas))
    axes = axes.flatten()

    for i, col in enumerate(columnas_categoricas):
        ax = axes[i]

        if relativa:
            total = df[col].value_counts().sum()
            serie = df[col].value_counts().apply(lambda x: x / total)
            sns.barplot(x=serie.index, y=serie, ax=ax, palette='viridis', hue=serie.index, legend=False)
            ax.set_ylabel('Frecuencia Relativa (%)')
        else:
            serie = df[col].value_counts()
            sns.barplot(x=serie.index, y=serie, ax=ax, palette='viridis', hue=serie.index, legend=False)
            ax.set_ylabel('Frecuencia')

        ax.set_title(f'Distribución de {col}')
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45)

        # Ajustar los límites del eje y para dar más espacio en la parte superior
        max_height = serie.max() if not relativa else 1.0  # Cambiar según si es frecuencia absoluta o relativa
        ax.set_ylim(0, max_height * 1.1)  # Aumentar el límite superior un 10%

        # Mostrar los valores encima de las barras
        if mostrar_valores:
            for p in ax.patches:
                height = p.get_height()
                if relativa:
                    percentage = height * 100  # Convertir a porcentaje
                    ax.annotate(f'{percentage:.0f}%', 
                                (p.get_x() + p.get_width() / 2., height),
                                ha='center', va='bottom',
                                xytext=(0, 5), textcoords='offset points')  # Ajustar el desplazamiento
                else:
                    ax.annotate(f'{int(height)}',
                                (p.get_x() + p.get_width() / 2., height),
                                ha='center', va='bottom',
                                xytext=(0, 5), textcoords='offset points')  # Ajustar el desplazamiento

    # Ocultar ejes no utilizados
    for j in range(i + 1, num_filas * 2):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

def CLASI_BI_FeaCat_bar_plot(df, target, features_to_compare, num_graf_per_line=2, show_percentage=False):
    num_features = len(features_to_compare)
    num_cols = num_graf_per_line
    num_rows = (num_features // num_cols) + (num_features % num_cols > 0)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    axes = axes.flatten()

    for i, feature in enumerate(features_to_compare):
        ax = axes[i]

        # Calcula el total de ocurrencias para cada combinación de target y feature
        total_counts = df.groupby([target])[feature].count()

        # Calcula los porcentajes
        percentage_df = df.groupby([target, feature]).size().reset_index(name='counts')
        percentage_df['percentage'] = percentage_df.apply(lambda row: (row['counts'] / total_counts[row[target]]) * 100, axis=1)

        # Genera una paleta de colores única para cada gráfico
        unique_hues = df[feature].nunique()
        palette = sns.color_palette("husl", unique_hues)

        # Dibuja el gráfico con porcentajes si show_percentage es True
        if show_percentage:
            sns.barplot(data=percentage_df, x=target, y='percentage', hue=feature, ax=ax, palette=palette)
            ax.set_ylabel('Porcentaje')
            for p in ax.patches:
                height = p.get_height()
                if height > 0:  # Solo coloca la etiqueta si la altura es mayor a cero
                    ax.text(p.get_x() + p.get_width() / 2., height + 0.5,
                            f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
        else:
            count_plot = sns.countplot(data=df, x=target, hue=feature, ax=ax, palette=palette)
            ax.set_ylabel('Conteo')
            for p in count_plot.patches:
                count = int(p.get_height())
                if count > 0:
                    x = p.get_x() + p.get_width() / 2
                    y = p.get_height()
                    ax.text(x, y + 0.5, str(count), ha='center', va='bottom', fontsize=10)

        ax.set_title(f'Conteo de {target} por {feature}')
        ax.set_xlabel(target)

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()



################
#   PRIVADAS   #
################

def _is_dataframe(df) -> bool:
    """
    Verifica si el objeto proporcionado es un DataFrame de pandas.

    Parámetros:
    -----------
    df : cualquier tipo
        Objeto que se desea verificar si es un DataFrame de pandas.

    Retorna:
    --------
    bool:
        Retorna `True` si el objeto es un DataFrame de pandas, de lo contrario, 
        imprime un mensaje de error y retorna `False`.

    Ejemplo:
    --------
    >>> _is_dataframe(pd.DataFrame())
    True
    
    >>> _is_dataframe([1, 2, 3])
    Error: Expected a pandas DataFrame
    False
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Expected a pandas DataFrame")
        return False
    else:
        return True
