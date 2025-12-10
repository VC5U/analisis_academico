import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
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

# ============================================================================
# FUNCIONES PARA CARGAR Y PREPARAR DATOS
# ============================================================================
@st.cache_data
def cargar_datos():
    """Cargar dataset y entender su estructura"""
    try:
        df = pd.read_csv('academic_performance_master.csv')
        
        # Información básica para debug
        st.session_state['dataset_info'] = {
            'filas': df.shape[0],
            'columnas': df.shape[1],
            'columnas_lista': df.columns.tolist()
        }
        
        return df
    except Exception as e:
        st.error(f"❌ Error al cargar dataset: {str(e)}")
        return None

@st.cache_data
def preparar_datos_para_modelos(df, limite_aprobacion=7.0):
    """Preparar datos para modelos ML"""
    df_clean = df.copy()
    
    # 1. Manejar valores nulos
    for col in df_clean.columns:
        if df_clean[col].isnull().sum() > 0:
            if df_clean[col].dtype == 'object':
                df_clean[col].fillna('Desconocido', inplace=True)
            else:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
    
    # 2. Crear variable objetivo (APROBADO/REPROBADO)
    if 'Nota_final' in df_clean.columns:
        # Verificar escala de notas
        nota_max = df_clean['Nota_final'].max()
        
        # Determinar escala (0-10 o 0-100)
        if nota_max <= 10:
            escala = "0-10"
            if limite_aprobacion > 10:
                limite_aprobacion = 7.0
                st.sidebar.info(f"📝 Notas en escala 0-10. Límite ajustado a {limite_aprobacion}")
        else:
            escala = "0-100"
        
        df_clean['Aprobado'] = (df_clean['Nota_final'] >= limite_aprobacion).astype(int)
        
        # Estadísticas
        aprobados = df_clean['Aprobado'].sum()
        total = len(df_clean)
        tasa_aprobacion = aprobados / total * 100
        
        return df_clean, 'Nota_final', limite_aprobacion, aprobados, total, tasa_aprobacion, escala
    else:
        st.error("❌ No se encontró columna 'Nota_final' en el dataset")
        return None, None, None, None, None, None, None

@st.cache_data
def crear_features_adicionales(df):
    """Crear características adicionales para mejorar modelos"""
    df_features = df.copy()
    
    # Si hay múltiples registros por estudiante, podemos agregar
    if 'Identificacion_Estudiante' in df.columns:
        stats_estudiante = df.groupby('Identificacion_Estudiante').agg({
            'Asistencia': 'mean',
            'Nota_final': 'mean',
            'Asignatura': 'count'
        }).rename(columns={
            'Asistencia': 'Asistencia_promedio',
            'Nota_final': 'Nota_promedio',
            'Asignatura': 'Num_asignaturas'
        }).reset_index()
        
        df_features = df_features.merge(stats_estudiante, 
                                      on='Identificacion_Estudiante', 
                                      how='left')
    
    # Codificar variables categóricas importantes
    categorical_cols = ['Nivel', 'Carrera']
    for col in categorical_cols:
        if col in df_features.columns:
            le = LabelEncoder()
            df_features[f'{col}_encoded'] = le.fit_transform(df_features[col].astype(str))
    
    return df_features

# ============================================================================
# SIDEBAR CONFIGURACIÓN
# ============================================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2232/2232688.png", width=100)
    st.title("🔍 Navegación")
    
    section = st.radio(
        "Selecciona una sección:",
        ["📊 Exploración de Datos", 
         "🤖 Modelo Supervisado", 
         "🔍 Modelo No Supervisado",
         "📈 Comparación",
         "🔮 Predicción"]
    )
    
    st.markdown("---")
    st.title("⚙️ Configuración")
    
    # Configuración general
    usar_features_adicionales = st.checkbox("Usar características adicionales", value=True)
    
    if section == "🤖 Modelo Supervisado":
        st.subheader("Configuración Modelo")
        test_size = st.slider("Tamaño conjunto prueba:", 0.1, 0.5, 0.3, 0.05)
        
        # Detectar escala de notas
        df_loaded = cargar_datos()
        if df_loaded is not None and 'Nota_final' in df_loaded.columns:
            nota_max = df_loaded['Nota_final'].max()
            if nota_max <= 10:
                limite_default = 7.0
                limite_min, limite_max, step = 0.0, 10.0, 0.5
            else:
                limite_default = 70.0
                limite_min, limite_max, step = 0.0, 100.0, 5.0
            
            limite_aprobacion = st.slider("Límite para aprobar:", 
                                         limite_min, limite_max, 
                                         limite_default, step)
        else:
            limite_aprobacion = st.slider("Límite para aprobar:", 0.0, 100.0, 70.0, 5.0)
        
    elif section == "🔍 Modelo No Supervisado":
        st.subheader("Configuración Clustering")
        n_clusters = st.slider("Número de clusters:", 2, 6, 3)
    
    st.markdown("---")
    st.title("📊 Información del Dataset")
    
    # Cargar y mostrar información del dataset
    df = cargar_datos()
    
    if df is not None:
        # Filtrar carreras no deseadas
        if 'Carrera' in df.columns:
            # Contar antes de filtrar
            total_antes = len(df)
            
            # Filtrar carreras no deseadas
            mascara_carreras = df['Carrera'].astype(str).str.startswith(('PREPARAREC', 'NT', 'CENTRO DE IDIOM'))
            df = df[~mascara_carreras].copy()
            
            total_despues = len(df)
            eliminados = total_antes - total_despues
            
            if eliminados > 0:
                st.success(f"✅ Dataset filtrado: {total_despues:,} registros")
                st.info(f"🗑️ Se eliminaron {eliminados:,} registros de carreras no deseadas")
            else:
                st.success(f"✅ Dataset: {total_despues:,} registros")
        else:
            st.success(f"✅ Dataset: {len(df):,} registros")
        
        # Información básica
        with st.expander("Ver detalles del dataset"):
            st.write("**Columnas disponibles:**")
            st.write(df.columns.tolist())
            
            if 'Nota_final' in df.columns:
                st.write("**Estadísticas de Nota_final:**")
                st.write(df['Nota_final'].describe())
                
                # Detectar escala
                nota_max = df['Nota_final'].max()
                if nota_max <= 10:
                    st.info("📝 **Escala detectada:** 0-10")
                else:
                    st.info("📝 **Escala detectada:** 0-100")
    else:
        st.error("No se pudo cargar el dataset")

# ============================================================================
# CARGAR Y PREPARAR DATOS
# ============================================================================
if 'df' not in locals() or df is None:
    df = cargar_datos()

if df is not None:
    # Aplicar filtro de carreras
    if 'Carrera' in df.columns:
        df = df[~df['Carrera'].astype(str).str.startswith(('PREPARAREC', 'NT', 'CENTRO DE IDIOM'))].copy()
    
    # Preparar datos según configuración
    if section == "🤖 Modelo Supervisado":
        df_clean, nota_col, limite, aprobados, total, tasa, escala = preparar_datos_para_modelos(
            df, limite_aprobacion if 'limite_aprobacion' in locals() else 7.0
        )
    else:
        df_clean, nota_col, limite, aprobados, total, tasa, escala = preparar_datos_para_modelos(df, 7.0)
    
    # Crear características adicionales si está habilitado
    if usar_features_adicionales and df_clean is not None:
        df_clean = crear_features_adicionales(df_clean)
else:
    st.error("❌ No se pudo cargar el dataset")
    st.stop()

# ============================================================================
# SECCIÓN 1: EXPLORACIÓN DE DATOS
# ============================================================================
if section == "📊 Exploración de Datos":
    st.header("📊 Exploración del Dataset")
    
    if df is not None:
        # Pestañas
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Vista General", "📈 Análisis Estadístico", 
                                          "🎓 Análisis Académico", "🔍 Calidad de Datos"])
        
        with tab1:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Primeros registros")
                st.dataframe(df.head(10), use_container_width=True, height=350)
                
                st.subheader("Resumen por estudiante")
                if 'Identificacion_Estudiante' in df.columns:
                    estudiantes_unicos = df['Identificacion_Estudiante'].nunique()
                    asignaturas_por_est = df.groupby('Identificacion_Estudiante')['Asignatura'].count()
                    
                    col_est1, col_est2, col_est3 = st.columns(3)
                    with col_est1:
                        st.metric("Estudiantes únicos", estudiantes_unicos)
                    with col_est2:
                        st.metric("Registros totales", len(df))
                    with col_est3:
                        st.metric("Prom. asignaturas/est", f"{asignaturas_por_est.mean():.1f}")
            
            with col2:
                st.subheader("Información General")
                st.metric("Total Registros", len(df))
                st.metric("Total Columnas", len(df.columns))
                
                if 'Nota_final' in df.columns:
                    nota_prom = df['Nota_final'].mean()
                    st.metric("Nota Promedio", f"{nota_prom:.2f}")
                    
                    # Distribución de estados
                    if 'Estado_Asignatura' in df.columns:
                        estados = df['Estado_Asignatura'].value_counts()
                        st.write("**Estado de Asignaturas:**")
                        for estado, count in estados.items():
                            porcentaje = count/len(df)*100
                            st.write(f"- {estado}: {count} ({porcentaje:.1f}%)")
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                # Histograma de notas
                if 'Nota_final' in df.columns:
                    fig_notas = px.histogram(df, x='Nota_final', nbins=30,
                                            title='Distribución de Notas Finales',
                                            color_discrete_sequence=['#636EFA'])
                    fig_notas.update_layout(xaxis_title="Nota Final", yaxis_title="Frecuencia")
                    st.plotly_chart(fig_notas, use_container_width=True)
                    
                    # Estadísticas
                    st.subheader("Estadísticas de Notas")
                    stats_df = df['Nota_final'].describe()
                    st.dataframe(pd.DataFrame(stats_df).T, use_container_width=True)
            
            with col2:
                # Boxplot de asistencia
                if 'Asistencia' in df.columns:
                    fig_asist = px.box(df, y='Asistencia', 
                                      title='Distribución de Asistencia',
                                      color_discrete_sequence=['#00CC96'])
                    st.plotly_chart(fig_asist, use_container_width=True)
                    
                    # Relación asistencia-nota (SIN trendline para evitar error)
                    if 'Nota_final' in df.columns:
                        fig_rel = px.scatter(df, x='Asistencia', y='Nota_final',
                                           title='Relación: Asistencia vs Nota Final',
                                           opacity=0.6,
                                           trendline=None)  # Sin trendline
                        st.plotly_chart(fig_rel, use_container_width=True)
                        
                        # Calcular correlación manualmente
                        correlacion = df['Asistencia'].corr(df['Nota_final'])
                        st.info(f"**Correlación Asistencia-Nota:** {correlacion:.3f}")
            
            # Matriz de correlación
            st.subheader("Matriz de Correlación")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                
                fig_corr = px.imshow(corr_matrix, 
                                    text_auto='.2f',
                                    aspect="auto",
                                    color_continuous_scale='RdBu',
                                    title='Correlación entre Variables Numéricas')
                st.plotly_chart(fig_corr, use_container_width=True)
        
        with tab3:
            st.subheader("🎓 Análisis Académico Detallado")
            
            # Análisis por carrera
            if 'Carrera' in df.columns:
                st.write("**Desempeño por Carrera (Top 15):**")
                carrera_stats = df.groupby('Carrera').agg({
                    'Nota_final': ['mean', 'count'],
                    'Asistencia': 'mean'
                }).round(2)
                
                carrera_stats.columns = ['Nota_promedio', 'Num_registros', 'Asistencia_promedio']
                
                # Ordenar y mostrar
                carrera_stats = carrera_stats.sort_values('Nota_promedio', ascending=False).head(15)
                st.dataframe(carrera_stats, use_container_width=True)
                
                # Gráfico de barras por carrera
                fig_carrera = px.bar(carrera_stats.reset_index(),
                                    x='Carrera', y='Nota_promedio',
                                    title='Top 15 Carreras por Nota Promedio',
                                    color='Nota_promedio',
                                    color_continuous_scale='Viridis')
                fig_carrera.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_carrera, use_container_width=True)
            
            # Análisis por nivel
            if 'Nivel' in df.columns:
                st.write("**Desempeño por Nivel Académico:**")
                nivel_stats = df.groupby('Nivel').agg({
                    'Nota_final': 'mean',
                    'Asistencia': 'mean',
                    'Identificacion_Estudiante': 'nunique'
                }).round(2)
                
                nivel_stats.columns = ['Nota_promedio', 'Asistencia_promedio', 'Estudiantes_unicos']
                st.dataframe(nivel_stats, use_container_width=True)
        
        with tab4:
            col1, col2 = st.columns(2)
            
            with col1:
                # Valores nulos
                st.subheader("Valores Nulos")
                null_counts = df.isnull().sum()
                null_df = pd.DataFrame({
                    'Columna': null_counts.index,
                    'Valores Nulos': null_counts.values,
                    '% Nulos': (null_counts.values / len(df) * 100).round(2)
                })
                null_df = null_df[null_df['Valores Nulos'] > 0]
                
                if len(null_df) > 0:
                    st.dataframe(null_df, use_container_width=True)
                    st.warning(f"⚠️ {len(null_df)} columnas tienen valores nulos")
                else:
                    st.success("✅ No hay valores nulos")
            
            with col2:
                # Duplicados
                st.subheader("Registros Duplicados")
                dup_count = df.duplicated().sum()
                
                if dup_count > 0:
                    st.error(f"❌ {dup_count} registros duplicados")
                else:
                    st.success("✅ No hay duplicados")
                
                # Valores únicos
                st.subheader("Valores Únicos por Columna")
                unique_counts = df.nunique()
                unique_df = pd.DataFrame({
                    'Columna': unique_counts.index,
                    'Valores Únicos': unique_counts.values
                }).sort_values('Valores Únicos', ascending=False).head(10)
                st.dataframe(unique_df, use_container_width=True)
    
    else:
        st.error("No hay datos para mostrar")

# ============================================================================
# SECCIÓN 2: MODELO SUPERVISADO
# ============================================================================
elif section == "🤖 Modelo Supervisado":
    st.header("🤖 Modelo de Clasificación Supervisada")
    
    if df_clean is not None:
        # Mostrar información
        st.info(f"""
        **📊 Información del Dataset Preparado:**
        - Total registros: {total:,}
        - Aprobados: {aprobados:,} ({tasa:.1f}%)
        - Límite de aprobación: {limite}
        - Escala de notas: {escala}
        """)
        
        # Verificar que tenemos ambas clases
        if df_clean['Aprobado'].nunique() < 2:
            st.error(f"""
            ⚠️ **PROBLEMA**: Solo hay una clase en los datos
            
            **Solución:**
            1. Ajusta el límite de aprobación en la barra lateral
            2. Actualmente usando límite: {limite}
            3. Rango de notas: {df_clean['Nota_final'].min():.1f} - {df_clean['Nota_final'].max():.1f}
            """)
            
            # Mostrar distribución de notas
            fig_dist = px.histogram(df_clean, x='Nota_final', nbins=30,
                                   title=f'Distribución de Notas (Límite: {limite})',
                                   color_discrete_sequence=['#FF6B6B'])
            fig_dist.add_vline(x=limite, line_dash="dash", line_color="green",
                              annotation_text=f"Límite: {limite}")
            st.plotly_chart(fig_dist, use_container_width=True)
            st.stop()
        
        # Preparar características
        st.subheader("🎯 Selección de Características")
        
        numeric_features = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        exclude_features = ['Aprobado', 'Nota_final']
        if 'Nota_promedio' in numeric_features:
            exclude_features.append('Nota_promedio')
        
        features = [col for col in numeric_features if col not in exclude_features]
        
        if len(features) == 0:
            st.error("No hay características numéricas disponibles")
            st.stop()
        
        st.success(f"✅ {len(features)} características seleccionadas")
        
        # Dividir datos
        X = df_clean[features]
        y = df_clean['Aprobado']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Estandarizar
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Entrenar modelo
        st.subheader("🚀 Entrenamiento del Modelo")
        
        with st.spinner("Entrenando modelo..."):
            try:
                model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
                model.fit(X_train_scaled, y_train)
                
                # Predicciones
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                
                # Métricas
                accuracy = accuracy_score(y_test, y_pred)
                conf_matrix = confusion_matrix(y_test, y_pred)
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                
                st.success("✅ Modelo entrenado exitosamente!")
                
                # Mostrar métricas
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Accuracy", f"{accuracy:.2%}")
                
                with col2:
                    precision = conf_matrix[1,1] / (conf_matrix[1,1] + conf_matrix[0,1]) if (conf_matrix[1,1] + conf_matrix[0,1]) > 0 else 0
                    st.metric("Precisión", f"{precision:.2%}")
                
                with col3:
                    recall = conf_matrix[1,1] / (conf_matrix[1,1] + conf_matrix[1,0]) if (conf_matrix[1,1] + conf_matrix[1,0]) > 0 else 0
                    st.metric("Recall", f"{recall:.2%}")
                
                with col4:
                    st.metric("ROC-AUC", f"{roc_auc:.3f}")
                
                # Matriz de confusión
                st.subheader("📊 Matriz de Confusión")
                
                fig_cm = px.imshow(conf_matrix,
                                  text_auto=True,
                                  color_continuous_scale='Blues',
                                  labels=dict(x="Predicción", y="Real", color="Cantidad"),
                                  x=['Reprobado', 'Aprobado'],
                                  y=['Reprobado', 'Aprobado'],
                                  title=f'Accuracy: {accuracy:.2%}')
                st.plotly_chart(fig_cm, use_container_width=True)
                
                # Reporte
                st.subheader("📋 Reporte de Clasificación")
                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df, use_container_width=True)
                
                # Importancia de características
                st.subheader("🔝 Importancia de Características")
                
                if hasattr(model, 'coef_'):
                    importance = pd.DataFrame({
                        'Característica': features,
                        'Importancia': np.abs(model.coef_[0])
                    }).sort_values('Importancia', ascending=False)
                    
                    fig_imp = px.bar(importance.head(15), 
                                    x='Importancia', 
                                    y='Característica',
                                    orientation='h',
                                    title='Top 15 Características Más Importantes',
                                    color='Importancia',
                                    color_continuous_scale='Viridis')
                    st.plotly_chart(fig_imp, use_container_width=True)
                    
                    st.dataframe(importance, use_container_width=True)
                
                # Curva ROC
                st.subheader("📈 Curva ROC")
                
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', 
                                            name=f'ROC (AUC = {roc_auc:.3f})',
                                            line=dict(color='blue', width=2)))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                            name='Línea base', line=dict(dash='dash', color='gray')))
                
                fig_roc.update_layout(title='Curva ROC',
                                     xaxis_title='Tasa de Falsos Positivos',
                                     yaxis_title='Tasa de Verdaderos Positivos')
                
                st.plotly_chart(fig_roc, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error al entrenar modelo: {str(e)}")
    else:
        st.error("No hay datos preparados para el modelo")

# ============================================================================
# SECCIÓN 3: MODELO NO SUPERVISADO
# ============================================================================
elif section == "🔍 Modelo No Supervisado":
    st.header("🔍 Clustering de Estudiantes")
    
    if df_clean is not None:
        st.info(f"📊 Dataset: {len(df_clean):,} registros")
        
        # Seleccionar características para clustering
        available_features = ['Asistencia', 'Nota_final']
        X_cluster = df_clean[available_features].copy()
        X_cluster = X_cluster.dropna()
        
        if len(X_cluster) < n_clusters:
            st.error(f"No hay suficientes datos ({len(X_cluster)}) para {n_clusters} clusters")
            st.stop()
        
        # Método del codo
        st.subheader("📉 Método del Codo")
        
        inertias = []
        k_range = range(1, 11)
        
        for k in k_range:
            kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans_temp.fit(X_cluster)
            inertias.append(kmeans_temp.inertia_)
        
        fig_elbow = px.line(x=list(k_range), y=inertias,
                           title='Método del Codo - Inercia vs Número de Clusters',
                           labels={'x': 'Número de Clusters (K)', 'y': 'Inercia'},
                           markers=True)
        fig_elbow.add_vline(x=n_clusters, line_dash="dash", line_color="red",
                           annotation_text=f"K seleccionado = {n_clusters}")
        st.plotly_chart(fig_elbow, use_container_width=True)
        
        # Aplicar K-means
        st.subheader(f"🎨 Visualización de Clusters (K={n_clusters})")
        
        with st.spinner(f"Aplicando K-means..."):
            scaler_cluster = StandardScaler()
            X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_cluster_scaled)
            
            df_viz = pd.DataFrame({
                'Asistencia': X_cluster['Asistencia'],
                'Nota_final': X_cluster['Nota_final'],
                'Cluster': clusters
            })
        
        # Gráfico de clusters
        fig_clusters = px.scatter(df_viz, x='Asistencia', y='Nota_final',
                                 color='Cluster', 
                                 title='Clustering: Asistencia vs Nota Final',
                                 color_continuous_scale='viridis',
                                 opacity=0.7)
        
        # Añadir centroides
        centroids_descaled = scaler_cluster.inverse_transform(kmeans.cluster_centers_)
        centroids_df = pd.DataFrame(centroids_descaled, columns=available_features)
        
        fig_clusters.add_trace(go.Scatter(
            x=centroids_df['Asistencia'],
            y=centroids_df['Nota_final'],
            mode='markers',
            marker=dict(symbol='x', size=15, color='red', line=dict(width=2)),
            name='Centroides'
        ))
        
        st.plotly_chart(fig_clusters, use_container_width=True)
        
        # Estadísticas por cluster
        st.subheader("📊 Estadísticas por Cluster")
        
        if 'Aprobado' in df_clean.columns:
            df_viz = df_viz.merge(df_clean[['Aprobado']], left_index=True, right_index=True)
            stats_cols = ['Asistencia', 'Nota_final', 'Aprobado']
        else:
            stats_cols = ['Asistencia', 'Nota_final']
        
        cluster_stats = df_viz.groupby('Cluster')[stats_cols].agg(['mean', 'std', 'count']).round(2)
        
        # Formatear tabla
        stats_display = pd.DataFrame()
        for col in stats_cols:
            for stat in ['mean', 'std']:
                if (col, stat) in cluster_stats.columns:
                    stats_display[f'{col}_{stat}'] = cluster_stats[(col, stat)]
        
        st.dataframe(stats_display, use_container_width=True)
        
        # Interpretación
        st.subheader("👥 Interpretación de Clusters")
        
        cluster_counts = df_viz['Cluster'].value_counts().sort_index()
        
        for cluster_id in range(n_clusters):
            with st.expander(f"Cluster {cluster_id} - {cluster_counts.get(cluster_id, 0)} estudiantes"):
                cluster_data = df_viz[df_viz['Cluster'] == cluster_id]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Características promedio:**")
                    asist_prom = cluster_data['Asistencia'].mean()
                    nota_prom = cluster_data['Nota_final'].mean()
                    st.write(f"• **Asistencia**: {asist_prom:.1f}%")
                    st.write(f"• **Nota final**: {nota_prom:.2f}")
                
                with col2:
                    if 'Aprobado' in cluster_data.columns:
                        aprob_rate = cluster_data['Aprobado'].mean() * 100
                        st.metric("Tasa de Aprobación", f"{aprob_rate:.1f}%")
                
                # Determinar perfil
                if 'Nota_final' in cluster_data.columns:
                    nota_prom = cluster_data['Nota_final'].mean()
                    
                    if nota_prom >= 8.5:
                        st.success("**🎯 PERFIL: EXCELENTES** - Alto rendimiento")
                    elif nota_prom >= 7.0:
                        st.info("**📚 PERFIL: BUENOS** - Rendimiento satisfactorio")
                    elif nota_prom >= 6.0:
                        st.warning("**⚠️ PERFIL: REGULARES** - Necesita mejora")
                    else:
                        st.error("**🚨 PERFIL: CRÍTICOS** - Intervención urgente")
    
    else:
        st.error("No hay datos para clustering")

# ============================================================================
# SECCIÓN 4: COMPARACIÓN
# ============================================================================
elif section == "📈 Comparación":
    st.header("📈 Comparación de Modelos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🤖 Modelo Supervisado")
        st.markdown("""
        ### ✅ **Fortalezas:**
        - **Alta precisión predictiva** para clasificación binaria
        - **Interpretación directa** de variables importantes  
        - **Probabilidades específicas** por estudiante
        - **Ideal para intervenciones** tempranas y personalizadas
        
        ### ⚠️ **Limitaciones:**
        - Requiere **datos etiquetados** previamente
        - Asume **relación lineal** entre variables
        - Sensible a **desbalance** de clases
        - Puede **sobreajustarse** a patrones históricos
        
        ### 🎯 **Mejor uso:**
        **Predicción individualizada** de riesgo académico
        """)
    
    with col2:
        st.subheader("🔍 Modelo No Supervisado")
        st.markdown("""
        ### ✅ **Fortalezas:**
        - **Descubre patrones ocultos** sin necesidad de etiquetas
        - **Identifica perfiles naturales** de estudiantes
        - **Útil para segmentación** y personalización de estrategias
        - **Detecta outliers** y casos atípicos automáticamente
        
        ### ⚠️ **Limitaciones:**
        - **Difícil evaluación objetiva** de resultados
        - **Sensible a selección** de características
        - **Requiere interpretación** experta de clusters
        - Necesita **definir número** de clusters manualmente
        
        ### 🎯 **Mejor uso:**
        **Segmentación estratégica** para pedagogía diferenciada
        """)
    
    st.markdown("---")
    
    # Integración recomendada
    st.subheader("🚀 Integración Recomendada")
    
    st.info("""
    ### **Estrategia combinada para máxima efectividad:**
    
    1. **Primero usar Clustering** para identificar **grupos naturales** de estudiantes
    2. **Luego aplicar Clasificación** dentro de cada grupo para **predecir riesgo específico**
    3. **Diseñar intervenciones personalizadas** según el **grupo + riesgo predicho**
    
    ### **Ejemplo de aplicación práctica:**
    
    | Cluster | Perfil | Estrategia Recomendada |
    |---------|---------|-----------------------|
    | **0** | 🎯 **Destacados** | Mentoría avanzada, oportunidades investigación |
    | **1** | 📚 **Regulares** | Refuerzo específico, seguimiento regular |
    | **2** | ⚠️ **En Riesgo** | Tutorías intensivas, plan de mejora |
    | **3** | 🚨 **Críticos** | Intervención inmediata, apoyo integral |
    """)

# ============================================================================
# SECCIÓN 5: PREDICCIÓN
# ============================================================================
else:
    st.header("🔮 Predicción Individual")
    
    if df_clean is not None and 'Aprobado' in df_clean.columns:
        # Entrenar modelo rápido para predicción
        numeric_features = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        exclude_features = ['Aprobado', 'Nota_final']
        if 'Nota_promedio' in numeric_features:
            exclude_features.append('Nota_promedio')
        
        features_pred = [col for col in numeric_features if col not in exclude_features]
        
        if len(features_pred) == 0:
            st.error("No hay características para entrenar modelo predictivo")
            st.stop()
        
        X_pred = df_clean[features_pred]
        y_pred = df_clean['Aprobado']
        
        # Verificar que tenemos ambas clases
        if y_pred.nunique() < 2:
            st.error("No hay suficientes clases para entrenar modelo predictivo")
            st.info("Ajusta el límite de aprobación en la sección 'Modelo Supervisado'")
            st.stop()
        
        # Entrenar modelo
        scaler_pred = StandardScaler()
        X_pred_scaled = scaler_pred.fit_transform(X_pred)
        
        model_pred = LogisticRegression(random_state=42, max_iter=1000)
        model_pred.fit(X_pred_scaled, y_pred)
        
        col_input, col_result = st.columns([1, 1])
        
        with col_input:
            st.subheader("📝 Características del Estudiante")
            
            # Inputs basados en características disponibles
            inputs = {}
            
            if 'Asistencia' in features_pred:
                asistencia = st.slider("Asistencia (%)", 0, 100, 85)
                inputs['Asistencia'] = asistencia
            
            if 'Participacion_clase' in features_pred:
                participacion = st.slider("Participación (0-10)", 0, 10, 7)
                inputs['Participacion_clase'] = participacion
            
            # Botón para predecir
            if st.button("🎯 Predecir Resultado", type="primary", use_container_width=True):
                # Crear datos del estudiante
                estudiante_data = {}
                for feature in features_pred:
                    if feature in inputs:
                        estudiante_data[feature] = inputs[feature]
                    else:
                        estudiante_data[feature] = df_clean[feature].median()
                
                estudiante_df = pd.DataFrame([estudiante_data])
                estudiante_df = estudiante_df[features_pred]
                
                # Estandarizar y predecir
                estudiante_scaled = scaler_pred.transform(estudiante_df)
                probabilidad = model_pred.predict_proba(estudiante_scaled)[0]
                prediccion = model_pred.predict(estudiante_scaled)[0]
                
                # Guardar resultados
                st.session_state['prediccion_resultados'] = {
                    'probabilidad': probabilidad,
                    'prediccion': prediccion,
                    'caracteristicas': inputs
                }
        
        with col_result:
            st.subheader("📊 Resultado de Predicción")
            
            if 'prediccion_resultados' in st.session_state:
                resultados = st.session_state['prediccion_resultados']
                
                # Calcular probabilidades
                prob_reprobado = resultados['probabilidad'][0] * 100
                prob_aprobado = resultados['probabilidad'][1] * 100
                
                if resultados['prediccion'] == 1:
                    st.success(f"""
                    ## ✅ **APROBADO**
                    
                    **Probabilidad de aprobar:** {prob_aprobado:.1f}%
                    
                    El estudiante tiene **alta probabilidad** de aprobar.
                    """)
                    st.balloons()
                else:
                    st.error(f"""
                    ## ❌ **REPROBADO**
                    
                    **Probabilidad de reprobar:** {prob_reprobado:.1f}%
                    
                    El estudiante tiene **alta probabilidad** de reprobar.
                    **Se recomienda intervención inmediata.**
                    """)
                
                # Gráfico de probabilidades
                fig_pred = px.bar(x=['Reprobado', 'Aprobado'], 
                                y=[resultados['probabilidad'][0], resultados['probabilidad'][1]],
                                color=['Reprobado', 'Aprobado'],
                                color_discrete_map={'Reprobado': '#EF553B', 'Aprobado': '#00CC96'},
                                labels={'x': 'Resultado', 'y': 'Probabilidad'},
                                title='Distribución de Probabilidades',
                                text=[f'{prob_reprobado:.1f}%', f'{prob_aprobado:.1f}%'])
                fig_pred.update_traces(textposition='outside')
                st.plotly_chart(fig_pred, use_container_width=True)
                
                # Recomendaciones
                st.subheader("📋 Recomendaciones")
                
                if resultados['prediccion'] == 0:
                    st.warning("""
                    **🔴 ACCIONES RECOMENDADAS (URGENTE):**
                    
                    1. **📅 Revisar asistencia** - Asegurar mínimo 80% de asistencia
                    2. **📚 Entrega de tareas** - Completar asignaciones pendientes
                    3. **👨‍🏫 Tutorías** - Solicitar sesiones de refuerzo inmediatas
                    4. **⏰ Horas de estudio** - Incrementar horas de estudio
                    5. **📊 Seguimiento** - Evaluación de progreso en 2 semanas
                    """)
                else:
                    st.info("""
                    **🟢 ACCIONES DE MANTENIMIENTO:**
                    
                    1. **✅ Continuar buen desempeño** - Mantener hábitos de estudio
                    2. **💬 Participación activa** - Seguir participando en clases
                    3. **🤝 Ayuda a compañeros** - Ofrecer apoyo a estudiantes
                    4. **🎯 Explorar profundización** - Buscar temas avanzados
                    """)
            else:
                st.info("👈 Ajusta las características y presiona 'Predecir Resultado'")
    else:
        st.error("No hay datos preparados para predicción")

# ============================================================================
# PIE DE PÁGINA
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>📚 <b>Práctica de Aprendizaje Automático</b> - Modelos Supervisado y No Supervisado</p>
    <p>🎓 <b>Análisis Predictivo del Rendimiento Académico</b></p>
    <p>⚙️ Desarrollado con Python, Scikit-learn y Streamlit</p>
    <p>📅 Diciembre 2025</p>
</div>
""", unsafe_allow_html=True)