import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Análisis Académico - ML",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🎓 Análisis Predictivo del Rendimiento Académico")
st.markdown("""
**Modelos Supervisado y No Supervisado** para predecir y entender el rendimiento estudiantil
""")

# Sidebar para navegación
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2232/2232688.png", width=100)
    st.title("Navegación")
    
    section = st.radio(
        "Selecciona una sección:",
        ["📊 Exploración de Datos", 
         "🤖 Modelo Supervisado", 
         "🔍 Modelo No Supervisado",
         "📈 Comparación",
         "🔮 Predicción"]
    )
    
    st.markdown("---")
    st.markdown("### Configuración")
    
    if section == "🤖 Modelo Supervisado":
        test_size = st.slider("Tamaño del conjunto de prueba:", 0.1, 0.5, 0.3, 0.05)
        limite_aprobacion = st.slider("Límite para aprobar:", 50, 90, 70, 5)
        
    elif section == "🔍 Modelo No Supervisado":
        n_clusters = st.slider("Número de clusters:", 2, 6, 3)
        feature_x = st.selectbox("Variable X:", ["Asistencia", "Nota_final", "Tareas_entregadas"])
        feature_y = st.selectbox("Variable Y:", ["Nota_final", "Asistencia", "Tareas_entregadas"])

# Función para cargar y preparar datos
@st.cache_data
def cargar_y_preparar_datos(limite_aprobacion=70):
    try:
        df = pd.read_csv('academic_performance_master.csv')
        
        # Verificar columnas disponibles
        st.sidebar.info(f"📊 Dataset: {df.shape[0]} filas, {df.shape[1]} columnas")
        
        # Mostrar columnas reales en debug
        debug_cols = st.sidebar.checkbox("Mostrar columnas del dataset", value=False)
        if debug_cols:
            st.sidebar.write("Columnas disponibles:", list(df.columns))
        
        # Crear copia para limpieza
        df_clean = df.copy()
        
        # Manejar valores nulos
        for col in df_clean.columns:
            if df_clean[col].isnull().sum() > 0:
                if df_clean[col].dtype == 'object':
                    df_clean[col].fillna('Desconocido', inplace=True)
                else:
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
        
        # Buscar columna de nota (puede tener diferentes nombres)
        nota_cols = [col for col in df_clean.columns if 'nota' in col.lower() or 'Nota' in col]
        if nota_cols:
            nota_col = nota_cols[0]
            st.sidebar.success(f"✅ Columna de nota encontrada: '{nota_col}'")
            
            # Crear variable objetivo
            df_clean['Aprobado'] = (df_clean[nota_col] >= limite_aprobacion).astype(int)
            
            # Verificar distribución de clases
            aprobados = df_clean['Aprobado'].sum()
            total = len(df_clean)
            st.sidebar.info(f"📊 Distribución: {aprobados} aprobados ({aprobados/total*100:.1f}%)")
            
            if aprobados == 0 or aprobados == total:
                st.sidebar.warning(f"⚠️ Solo hay una clase en los datos. Ajusta el límite de aprobación.")
                
        else:
            st.sidebar.error("❌ No se encontró columna de nota en el dataset")
            # Crear variable objetivo ficticia para continuar
            df_clean['Aprobado'] = np.random.choice([0, 1], size=len(df_clean), p=[0.3, 0.7])
        
        return df, df_clean
        
    except Exception as e:
        st.sidebar.error(f"❌ Error al cargar datos: {str(e)}")
        
        # Crear datos de ejemplo
        np.random.seed(42)
        n_estudiantes = 200
        
        datos = {
            'Estudiante': [f'EST{i:03d}' for i in range(n_estudiantes)],
            'Nombre': [f'Estudiante_{i}' for i in range(n_estudiantes)],
            'Asistencia': np.random.normal(85, 10, n_estudiantes).clip(60, 100).astype(int),
            'Tareas_entregadas': np.random.randint(5, 20, n_estudiantes),
            'Participacion_clase': np.random.normal(7, 2, n_estudiantes).clip(0, 10).astype(int),
            'Horas_estudio': np.random.normal(12, 4, n_estudiantes).clip(2, 25).astype(int),
            'Nota_parcial1': np.random.normal(75, 15, n_estudiantes).clip(30, 100).astype(int),
            'Nota_parcial2': np.random.normal(72, 18, n_estudiantes).clip(30, 100).astype(int),
            'Nota_final': np.random.normal(70, 20, n_estudiantes).clip(0, 100).astype(int),
            'Nivel': np.random.choice(['Licenciatura', 'Maestría'], n_estudiantes, p=[0.8, 0.2])
        }
        
        df = pd.DataFrame(datos)
        df_clean = df.copy()
        df_clean['Aprobado'] = (df_clean['Nota_final'] >= limite_aprobacion).astype(int)
        
        st.sidebar.warning("⚠️ Usando datos de ejemplo")
        return df, df_clean

# Cargar datos según la sección
if section == "🤖 Modelo Supervisado":
    limite = st.session_state.get('limite_aprobacion', 70)
    if 'limite_aprobacion' in st.session_state:
        limite = st.session_state.limite_aprobacion
    df, df_clean = cargar_y_preparar_datos(limite)
else:
    df, df_clean = cargar_y_preparar_datos()

# ============================================================================
# SECCIÓN 1: EXPLORACIÓN DE DATOS
# ============================================================================
if section == "📊 Exploración de Datos":
    st.header("📊 Exploración del Dataset")
    
    # Pestañas
    tab1, tab2, tab3 = st.tabs(["📋 Vista General", "📈 Análisis", "🔍 Calidad"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Primeros registros")
            st.dataframe(df.head(10), use_container_width=True, height=300)
            
            st.subheader("Últimos registros")
            st.dataframe(df.tail(5), use_container_width=True, height=200)
        
        with col2:
            st.subheader("Información general")
            st.metric("Total de estudiantes", len(df))
            st.metric("Variables", len(df.columns))
            
            if 'Aprobado' in df_clean.columns:
                aprobados = df_clean['Aprobado'].sum()
                total = len(df_clean)
                st.metric("Estudiantes que aprueban", aprobados)
                st.metric("Tasa de aprobación", f"{aprobados/total*100:.1f}%")
            
            # Mostrar tipos de datos
            st.subheader("Tipos de datos")
            tipos = pd.DataFrame(df.dtypes, columns=['Tipo'])
            st.dataframe(tipos, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Buscar columna numérica para histograma
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 0:
                col_selected = st.selectbox("Selecciona variable para histograma:", numeric_cols)
                
                fig = px.histogram(df, x=col_selected, nbins=30,
                                  title=f'Distribución de {col_selected}',
                                  color_discrete_sequence=['#636EFA'])
                fig.update_layout(xaxis_title=col_selected, yaxis_title="Frecuencia")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No hay columnas numéricas para analizar")
        
        with col2:
            # Boxplot
            if len(numeric_cols) > 0:
                col_box = st.selectbox("Selecciona variable para boxplot:", numeric_cols, 
                                      key='boxplot_select')
                
                fig = px.box(df, y=col_box, 
                            title=f'Boxplot de {col_box}',
                            color_discrete_sequence=['#00CC96'])
                st.plotly_chart(fig, use_container_width=True)
        
        # Matriz de correlación
        st.subheader("Matriz de Correlación")
        
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            
            fig = px.imshow(corr_matrix, 
                           text_auto='.2f',
                           aspect="auto",
                           color_continuous_scale='RdBu',
                           title='Correlación entre variables')
            st.plotly_chart(fig, use_container_width=True)
            
            # Top correlaciones
            st.subheader("Correlaciones más fuertes")
            if 'Nota_final' in corr_matrix.columns:
                correlaciones = corr_matrix['Nota_final'].abs().sort_values(ascending=False)
                correlaciones = correlaciones[correlaciones.index != 'Nota_final']
                
                top_corr = correlaciones.head(5)
                st.write(top_corr)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            # Valores nulos
            st.subheader("Valores Nulos")
            null_counts = df.isnull().sum()
            null_df = pd.DataFrame({
                'Variable': null_counts.index,
                'Valores nulos': null_counts.values,
                '% Nulos': (null_counts.values / len(df) * 100).round(2)
            })
            null_df = null_df[null_df['Valores nulos'] > 0]
            
            if len(null_df) > 0:
                st.dataframe(null_df, use_container_width=True)
            else:
                st.success("✅ No hay valores nulos")
        
        with col2:
            # Duplicados
            st.subheader("Registros Duplicados")
            dup_count = df.duplicated().sum()
            if dup_count > 0:
                st.warning(f"⚠️ {dup_count} registros duplicados encontrados")
                
                if st.button("Mostrar duplicados"):
                    duplicates = df[df.duplicated(keep=False)]
                    st.dataframe(duplicates, use_container_width=True)
            else:
                st.success("✅ No hay registros duplicados")
            
            # Valores únicos
            st.subheader("Valores Únicos por Columna")
            unique_counts = df.nunique()
            unique_df = pd.DataFrame({
                'Variable': unique_counts.index,
                'Valores únicos': unique_counts.values
            })
            st.dataframe(unique_df.head(10), use_container_width=True)

# ============================================================================
# SECCIÓN 2: MODELO SUPERVISADO
# ============================================================================
elif section == "🤖 Modelo Supervisado":
    st.header("🤖 Modelo de Clasificación Supervisada")
    
    # Actualizar límite en sesión
    st.session_state.limite_aprobacion = limite_aprobacion
    
    # Recargar datos con nuevo límite
    df, df_clean = cargar_y_preparar_datos(limite_aprobacion)
    
    # Verificar que tenemos ambas clases
    if 'Aprobado' not in df_clean.columns:
        st.error("No se pudo crear la variable objetivo 'Aprobado'")
        st.stop()
    
    clase_counts = df_clean['Aprobado'].value_counts()
    if len(clase_counts) < 2:
        st.warning(f"""
        ⚠️ **Problema**: Solo hay una clase en los datos ({clase_counts.index[0]})
        
        **Causas posibles:**
        1. El límite de aprobación ({limite_aprobacion}) es muy alto/bajo
        2. Todos los estudiantes tienen notas similares
        3. El dataset tiene un desbalance extremo
        
        **Solución:**
        - Ajusta el límite de aprobación en la barra lateral
        - O usa datos de ejemplo (selecciona en barra lateral)
        """)
        
        # Mostrar estadísticas de notas
        nota_cols = [col for col in df_clean.columns if 'nota' in col.lower() or 'Nota' in col]
        if nota_cols:
            st.subheader("Estadísticas de Notas")
            nota_col = nota_cols[0]
            st.write(f"Columna de nota: {nota_col}")
            st.write(df_clean[nota_col].describe())
        
        st.stop()
    
    # Preparar datos para modelo
    st.subheader("Preparación de Datos")
    
    # Seleccionar características numéricas
    numeric_features = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    
    # Excluir columnas no relevantes
    exclude_features = ['Aprobado']
    for col in df_clean.columns:
        if 'nota' in col.lower() or 'Nota' in col:
            exclude_features.append(col)
    
    features = [col for col in numeric_features if col not in exclude_features]
    
    if len(features) == 0:
        st.error("No hay características numéricas para entrenar el modelo")
        st.stop()
    
    st.write(f"**Características seleccionadas:** {len(features)} variables")
    st.write(features)
    
    X = df_clean[features]
    y = df_clean['Aprobado']
    
    # Mostrar distribución de clases
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total muestras", len(X))
    with col2:
        st.metric("Clase 0 (Reprobados)", (y == 0).sum())
    with col3:
        st.metric("Clase 1 (Aprobados)", (y == 1).sum())
    
    # Estandarizar y dividir
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Entrenar modelo
    st.subheader("Entrenamiento del Modelo")
    
    try:
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        
        # Predicciones y métricas
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Mostrar resultados
        st.success(f"✅ Modelo entrenado exitosamente")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Accuracy", f"{accuracy:.2%}")
        
        with col2:
            precision = conf_matrix[1,1] / (conf_matrix[1,1] + conf_matrix[0,1]) if (conf_matrix[1,1] + conf_matrix[0,1]) > 0 else 0
            st.metric("Precisión", f"{precision:.2%}")
        
        with col3:
            recall = conf_matrix[1,1] / (conf_matrix[1,1] + conf_matrix[1,0]) if (conf_matrix[1,1] + conf_matrix[1,0]) > 0 else 0
            st.metric("Recall", f"{recall:.2%}")
        
        # Matriz de confusión
        st.subheader("Matriz de Confusión")
        fig = px.imshow(conf_matrix,
                       text_auto=True,
                       color_continuous_scale='Blues',
                       labels=dict(x="Predicción", y="Real", color="Cantidad"),
                       x=['Reprobado', 'Aprobado'],
                       y=['Reprobado', 'Aprobado'],
                       title=f'Accuracy: {accuracy:.2%}')
        st.plotly_chart(fig, use_container_width=True)
        
        # Reporte
        st.subheader("Reporte de Clasificación")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, use_container_width=True)
        
        # Importancia de características
        st.subheader("Importancia de Características")
        if hasattr(model, 'coef_'):
            importance = pd.DataFrame({
                'Variable': features,
                'Importancia': np.abs(model.coef_[0])
            }).sort_values('Importancia', ascending=False)
            
            fig = px.bar(importance, 
                        x='Importancia', 
                        y='Variable',
                        orientation='h',
                        title='Importancia de Variables',
                        color='Importancia',
                        color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(importance, use_container_width=True)
    
    except Exception as e:
        st.error(f"❌ Error al entrenar modelo: {str(e)}")

# ============================================================================
# SECCIÓN 3: MODELO NO SUPERVISADO
# ============================================================================
elif section == "🔍 Modelo No Supervisado":
    st.header("🔍 Clustering de Estudiantes")
    
    # Verificar características disponibles
    available_features = [col for col in ['Asistencia', 'Nota_final', 'Tareas_entregadas'] 
                         if col in df_clean.columns]
    
    if len(available_features) < 2:
        # Usar primeras columnas numéricas
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        available_features = numeric_cols[:min(3, len(numeric_cols))]
        st.warning(f"Usando características disponibles: {available_features}")
    
    if len(available_features) < 2:
        st.error("Se necesitan al menos 2 características numéricas para clustering")
        st.stop()
    
    # Verificar que las características seleccionadas existan
    if feature_x not in available_features:
        feature_x = available_features[0]
    if feature_y not in available_features:
        feature_y = available_features[1] if len(available_features) > 1 else available_features[0]
    
    # Seleccionar datos para clustering
    cluster_features = list(set([feature_x, feature_y]))
    X_cluster = df_clean[cluster_features].copy()
    
    # Eliminar valores nulos
    X_cluster = X_cluster.dropna()
    
    if len(X_cluster) < n_clusters:
        st.error(f"No hay suficientes datos ({len(X_cluster)}) para {n_clusters} clusters")
        st.stop()
    
    # Aplicar K-means
    scaler_cluster = StandardScaler()
    X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_cluster_scaled)
    
    df_clustered = df_clean.copy()
    # Asegurar que los índices coincidan
    df_clustered = df_clustered.loc[X_cluster.index].copy()
    df_clustered['Cluster'] = clusters
    
    # Método del codo
    st.subheader("Método del Codo para Determinar K Óptimo")
    inertias = []
    k_range = range(1, 11)
    
    for k in k_range:
        kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans_temp.fit(X_cluster_scaled)
        inertias.append(kmeans_temp.inertia_)
    
    fig1 = px.line(x=list(k_range), y=inertias, 
                  title='Método del Codo',
                  labels={'x': 'Número de Clusters', 'y': 'Inercia'},
                  markers=True)
    fig1.add_vline(x=n_clusters, line_dash="dash", line_color="red",
                  annotation_text=f"K seleccionado = {n_clusters}")
    st.plotly_chart(fig1, use_container_width=True)
    
    # Visualización de clusters
    st.subheader(f"Visualización de Clusters ({n_clusters} grupos)")
    
    # Preparar hover data
    hover_columns = []
    for col in df_clustered.columns:
        if col not in cluster_features + ['Cluster']:
            # Tomar solo algunas columnas para hover
            if len(hover_columns) < 3:  # Máximo 3 columnas adicionales
                hover_columns.append(col)
    
    fig2 = px.scatter(df_clustered, 
                     x=feature_x,
                     y=feature_y,
                     color='Cluster',
                     title=f'Clustering: {feature_x} vs {feature_y}',
                     hover_data=hover_columns[:3],  # Limitar a 3 columnas
                     color_continuous_scale='viridis')
    
    # Añadir centroides
    centroids_descaled = scaler_cluster.inverse_transform(kmeans.cluster_centers_)
    centroids_df = pd.DataFrame(centroids_descaled, columns=cluster_features)
    centroids_df['Cluster'] = range(n_clusters)
    
    fig2.add_trace(go.Scatter(
        x=centroids_df[feature_x],
        y=centroids_df[feature_y],
        mode='markers',
        marker=dict(symbol='x', size=15, color='red', line=dict(width=2)),
        name='Centroides',
        hoverinfo='skip'
    ))
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # Estadísticas por cluster
    st.subheader("Estadísticas por Cluster")
    
    # Seleccionar columnas para estadísticas
    stats_cols = cluster_features.copy()
    if 'Aprobado' in df_clustered.columns:
        stats_cols.append('Aprobado')
    
    cluster_stats = df_clustered.groupby('Cluster')[stats_cols].agg(['mean', 'std', 'count']).round(2)
    
    # Formatear mejor la tabla
    cluster_stats_flat = pd.DataFrame()
    for col in stats_cols:
        for stat in ['mean', 'std']:
            if (col, stat) in cluster_stats.columns:
                cluster_stats_flat[f'{col}_{stat}'] = cluster_stats[(col, stat)]
    
    st.dataframe(cluster_stats_flat, use_container_width=True)
    
    # Distribución de clusters
    st.subheader("Distribución de Estudiantes por Cluster")
    cluster_counts = df_clustered['Cluster'].value_counts().sort_index()
    
    fig3 = px.bar(x=cluster_counts.index.astype(str), 
                  y=cluster_counts.values,
                  title='Número de Estudiantes por Cluster',
                  labels={'x': 'Cluster', 'y': 'Cantidad de Estudiantes'},
                  color=cluster_counts.index.astype(str))
    st.plotly_chart(fig3, use_container_width=True)
    
    # Interpretación
    st.subheader("Interpretación de Clusters")
    
    for cluster_id in range(n_clusters):
        with st.expander(f"Cluster {cluster_id} - {cluster_counts.get(cluster_id, 0)} estudiantes"):
            cluster_data = df_clustered[df_clustered['Cluster'] == cluster_id]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Características promedio:**")
                for feature in stats_cols[:3]:  # Mostrar primeras 3
                    if feature in cluster_data.columns:
                        avg = cluster_data[feature].mean()
                        st.write(f"- {feature}: {avg:.2f}")
            
            with col2:
                if 'Aprobado' in cluster_data.columns:
                    aprob_rate = cluster_data['Aprobado'].mean() * 100
                    st.metric("Tasa de Aprobación", f"{aprob_rate:.1f}%")
            
            # Determinar perfil
            if 'Nota_final' in cluster_data.columns:
                avg_grade = cluster_data['Nota_final'].mean()
                if avg_grade >= 80:
                    st.success("🎯 **Estudiantes Destacados**: Alto rendimiento académico")
                elif avg_grade >= 70:
                    st.info("📚 **Estudiantes Regulares**: Rendimiento satisfactorio")
                elif avg_grade >= 60:
                    st.warning("⚠️ **Estudiantes en Riesgo**: Requieren atención")
                else:
                    st.error("🚨 **Estudiantes Críticos**: Necesitan intervención inmediata")

# ============================================================================
# SECCIÓN 4: COMPARACIÓN
# ============================================================================
elif section == "📈 Comparación":
    st.header("📈 Comparación de Modelos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔮 Modelo Supervisado")
        st.markdown("""
        ### ✅ Ventajas:
        - **Alta precisión predictiva** para clasificación
        - **Interpretación directa** de variables importantes
        - **Probabilidades específicas** por estudiante
        - **Útil para intervenciones tempranas** y personalizadas
        
        ### ⚠️ Limitaciones:
        - **Requiere datos etiquetados** previamente
        - **Asume relación lineal** entre variables
        - **Sensible a desbalance** de clases
        - **Puede sobreajustarse** a datos históricos
        
        ### 🎯 Mejor uso:
        **Predicción individualizada** de riesgo académico
        """)
    
    with col2:
        st.subheader("🔍 Modelo No Supervisado")
        st.markdown("""
        ### ✅ Ventajas:
        - **Descubre patrones ocultos** sin etiquetas previas
        - **Identifica perfiles naturales** de estudiantes
        - **Útil para segmentación** y personalización
        - **Detecta outliers** y casos atípicos
        
        ### ⚠️ Limitaciones:
        - **Difícil evaluación objetiva** de resultados
        - **Sensible a selección** de características
        - **Requiere interpretación** experta
        - **Necesita definir** número de clusters
        
        ### 🎯 Mejor uso:
        **Segmentación para estrategias** pedagógicas diferenciadas
        """)
    
    st.markdown("---")
    
    # Integración recomendada
    st.subheader("🚀 Integración Recomendada")
    
    st.info("""
    ### **Estrategia combinada para máxima efectividad:**
    
    1. **Primero usar Clustering** para identificar grupos naturales de estudiantes
    2. **Luego aplicar Clasificación** dentro de cada grupo para predecir riesgo específico
    3. **Diseñar intervenciones personalizadas** según el grupo y riesgo predicho
    
    ### **Ejemplo de aplicación:**
    
    | Cluster | Perfil | Estrategia Recomendada |
    |---------|---------|-----------------------|
    | 0 | 🎯 Destacados | Mentoría avanzada, oportunidades investigación |
    | 1 | 📚 Regulares | Refuerzo en áreas específicas, seguimiento regular |
    | 2 | ⚠️ En Riesgo | Tutorías intensivas, seguimiento cercano |
    | 3 | 🚨 Críticos | Intervención inmediata, apoyo integral |
    
    ### **Beneficios:**
    - **Mayor precisión**: Modelos específicos por grupo
    - **Intervenciones efectivas**: Estrategias personalizadas
    - **Uso eficiente de recursos**: Enfoque en quienes más lo necesitan
    - **Prevención temprana**: Identificación proactiva de riesgo
    """)

# ============================================================================
# SECCIÓN 5: PREDICCIÓN
# ============================================================================
else:
    st.header("🔮 Predicción Individual")
    
    # Verificar que tenemos datos con variable objetivo
    if 'Aprobado' not in df_clean.columns:
        st.error("No se puede realizar predicción - falta variable objetivo")
        st.info("Ve a la sección 'Modelo Supervisado' primero para configurar el límite de aprobación")
        st.stop()
    
    # Verificar que tenemos ambas clases
    if df_clean['Aprobado'].nunique() < 2:
        st.warning("No hay suficientes clases para entrenar modelo predictivo")
        st.info("Ajusta el límite de aprobación en la sección 'Modelo Supervisado'")
        st.stop()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Características del Estudiante")
        
        # Sliders con valores por defecto realistas
        asistencia = st.slider("Asistencia (%)", 0, 100, 85)
        tareas = st.slider("Tareas entregadas", 0, 30, 15)
        participacion = st.slider("Participación (0-10)", 0, 10, 7)
        horas_estudio = st.slider("Horas de estudio semanales", 0, 40, 12)
        
        # Botón para predecir
        predecir = st.button("🔮 Predecir Resultado", type="primary")
    
    with col2:
        # Preparar características para modelo
        numeric_features = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        
        # Excluir columnas no relevantes
        exclude_features = ['Aprobado']
        for col in df_clean.columns:
            if 'nota' in col.lower() or 'Nota' in col:
                exclude_features.append(col)
        
        features = [col for col in numeric_features if col not in exclude_features]
        
        if len(features) == 0:
            st.error("No hay características para entrenar modelo")
            st.stop()
        
        # Entrenar modelo con todas las características disponibles
        X = df_clean[features]
        y = df_clean['Aprobado']
        
        # Verificar que tenemos datos
        if len(X) == 0:
            st.error("No hay datos para entrenar el modelo")
            st.stop()
        
        # Estandarizar
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Entrenar modelo
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_scaled, y)
        
        # Crear datos del estudiante (asegurar todas las características)
        estudiante_data = {}
        for feature in features:
            if feature == 'Asistencia':
                estudiante_data[feature] = asistencia
            elif feature == 'Tareas_entregadas' or 'tarea' in feature.lower():
                estudiante_data[feature] = tareas
            elif 'participacion' in feature.lower():
                estudiante_data[feature] = participacion
            elif 'hora' in feature.lower() or 'estudio' in feature.lower():
                estudiante_data[feature] = horas_estudio
            else:
                # Para otras características, usar la mediana
                estudiante_data[feature] = df_clean[feature].median()
        
        estudiante_df = pd.DataFrame([estudiante_data])
        
        # Asegurar el mismo orden de columnas
        estudiante_df = estudiante_df[features]
        
        # Estandarizar
        estudiante_scaled = scaler.transform(estudiante_df)
        
        # Predecir solo cuando se presiona el botón
        if predecir:
            try:
                probabilidad = model.predict_proba(estudiante_scaled)[0]
                prediccion = model.predict(estudiante_scaled)[0]
                
                st.subheader("Resultado de la Predicción")
                
                # Mostrar resultado con estilo
                if prediccion == 1:
                    st.success("""
                    ## ✅ **APROBADO**
                    
                    **El estudiante tiene alta probabilidad de aprobar la asignatura.**
                    """)
                    st.balloons()
                else:
                    st.error("""
                    ## ❌ **REPROBADO**
                    
                    **El estudiante tiene alta probabilidad de reprobar la asignatura.**
                    **Se recomienda intervención inmediata.**
                    """)
                
                # Mostrar probabilidades
                col_prob1, col_prob2 = st.columns(2)
                with col_prob1:
                    st.metric("Probabilidad de Aprobar", f"{probabilidad[1]*100:.1f}%")
                with col_prob2:
                    st.metric("Probabilidad de Reprobado", f"{probabilidad[0]*100:.1f}%")
                
                # Gráfico de probabilidades
                fig = px.bar(x=['Reprobado', 'Aprobado'], 
                            y=[probabilidad[0], probabilidad[1]],
                            color=['Reprobado', 'Aprobado'],
                            color_discrete_map={'Reprobado': '#EF553B', 'Aprobado': '#00CC96'},
                            labels={'x': 'Resultado', 'y': 'Probabilidad'},
                            title='Distribución de Probabilidades',
                            text=[f'{probabilidad[0]*100:.1f}%', f'{probabilidad[1]*100:.1f}%'])
                fig.update_traces(textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
                
                # Recomendaciones basadas en predicción
                st.subheader("📋 Recomendaciones")
                
                if prediccion == 0:  # Si predice reprobado
                    st.warning("""
                    **Acciones recomendadas:**
                    
                    1. **Revisar asistencia**: Asegurar mínimo 80% de asistencia
                    2. **Entrega de tareas**: Completar todas las asignaciones pendientes
                    3. **Tutorías**: Solicitar sesiones de refuerzo con el docente
                    4. **Horas de estudio**: Incrementar a mínimo 15 horas semanales
                    5. **Seguimiento**: Programar evaluación de progreso en 2 semanas
                    """)
                else:
                    st.info("""
                    **Acciones de mantenimiento:**
                    
                    1. **Continuar con buen desempeño**: Mantener hábitos de estudio
                    2. **Participación activa**: Seguir participando en clases
                    3. **Ayuda a compañeros**: Ofrecer apoyo a estudiantes con dificultades
                    4. **Explorar profundización**: Buscar temas avanzados de interés
                    """)
                
                # Factores de influencia
                if hasattr(model, 'coef_'):
                    importancia = np.abs(model.coef_[0])
                    idx_importante = np.argmax(importancia)
                    variable_importante = features[idx_importante]
                    
                    st.info(f"""
                    **Factor más influyente en la predicción:**
                    ### **{variable_importante}**
                    
                    Mejorar en esta variable aumentaría significativamente 
                    las probabilidades de aprobar.
                    """)
            
            except Exception as e:
                st.error(f"❌ Error al realizar predicción: {str(e)}")
        
        else:
            # Mostrar placeholder cuando no se ha presionado el botón
            st.subheader("Resultado de la Predicción")
            st.info("👈 Ajusta las características del estudiante y presiona 'Predecir Resultado'")
            
            # Mostrar valores actuales
            st.write("**Valores actuales:**")
            st.write(f"- Asistencia: {asistencia}%")
            st.write(f"- Tareas entregadas: {tareas}")
            st.write(f"- Participación: {participacion}/10")
            st.write(f"- Horas de estudio: {horas_estudio} horas/semana")

# ============================================================================
# PIE DE PÁGINA
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>📚 Práctica de Aprendizaje Automático - Modelos Supervisado y No Supervisado</p>
    <p>🎓 Análisis Predictivo del Rendimiento Académico</p>
    <p>⚙️ Desarrollado con Python, Scikit-learn y Streamlit</p>
</div>
""", unsafe_allow_html=True)

# Debug info en sidebar (solo en desarrollo)
debug_mode = st.sidebar.checkbox("Modo debug", value=False)
if debug_mode:
    st.sidebar.write("### Debug Info")
    st.sidebar.write(f"Sección actual: {section}")
    st.sidebar.write(f"Tamaño dataset original: {df.shape if 'df' in locals() else 'N/A'}")
    st.sidebar.write(f"Tamaño dataset limpio: {df_clean.shape if 'df_clean' in locals() else 'N/A'}")
    
    if 'df_clean' in locals():
        st.sidebar.write("Columnas df_clean:", list(df_clean.columns))
        
        if 'Aprobado' in df_clean.columns:
            st.sidebar.write("Distribución de Aprobado:")
            st.sidebar.write(df_clean['Aprobado'].value_counts())