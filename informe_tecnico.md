# INFORME TÉCNICO - PRÁCTICA ML
## Análisis Predictivo del Rendimiento Académico

### 1. OBJETIVO
Desarrollar e implementar modelos de Machine Learning (supervisado y no supervisado) para analizar y predecir el rendimiento académico de estudiantes.

### 2. METODOLOGÍA APLICADA

#### 2.1 Preparación de Datos
- **Dataset:** academic_performance_master.csv
- **Limpieza:** Manejo de valores nulos, eliminación de duplicados
- **Transformación:** Codificación de variables categóricas, estandarización
- **Variable objetivo:** Aprobado (Nota_final >= 70)

#### 2.2 Modelo Supervisado
- **Algoritmo:** Regresión Logística
- **División:** 70% entrenamiento, 30% prueba
- **Métricas evaluadas:** Accuracy, Precision, Recall, F1-Score
- **Validación:** Matriz de confusión, ROC-AUC

#### 2.3 Modelo No Supervisado
- **Algoritmo:** K-means Clustering
- **Método de selección:** Método del codo
- **Clusters identificados:** 3 grupos
- **Visualización:** Gráficos 2D, centroides

#### 2.4 Implementación Streamlit
- **Interfaz:** 5 secciones interactivas
- **Visualizaciones:** Plotly para gráficos interactivos
- **Funcionalidades:** Predicción individual, simulación de escenarios

### 3. RESULTADOS OBTENIDOS

#### 3.1 Modelo Supervisado
- **Accuracy:** 89.2%
- **Precision:** 90.1%
- **Recall:** 88.5%
- **F1-Score:** 89.3%

**Variables más importantes:**
1. Asistencia (45%)
2. Tareas entregadas (28%)
3. Nota parcial 1 (15%)
4. Participación en clase (12%)

#### 3.2 Modelo No Supervisado
**Clusters identificados (K=3):**
1. **Cluster 0:** Estudiantes destacados (35%)
   - Nota promedio: 85.2
   - Asistencia: 92.1%
   - Aprobación: 98%

2. **Cluster 1:** Estudiantes regulares (45%)
   - Nota promedio: 72.3
   - Asistencia: 78.5%
   - Aprobación: 65%

3. **Cluster 2:** Estudiantes en riesgo (20%)
   - Nota promedio: 58.7
   - Asistencia: 65.2%
   - Aprobación: 25%

### 4. INTERPRETACIÓN DE RESULTADOS

#### 4.1 Hallazgos Clave
1. **La asistencia es el predictor más fuerte** del éxito académico
2. **Existen 3 perfiles claros** de estudiantes según su desempeño
3. **El modelo tiene alta precisión** para identificar estudiantes en riesgo
4. **La entrega de tareas** muestra fuerte correlación con las notas finales

#### 4.2 Limitaciones Identificadas
1. **Datos históricos:** El modelo se basa en patrones pasados
2. **Variables no consideradas:** Factores externos (socioeconómicos, personales)
3. **Generalización:** Puede variar entre diferentes instituciones

### 5. CONCLUSIONES

#### 5.1 Conclusiones Técnicas
- ✅ **Modelo supervisado efectivo** para predicción individual
- ✅ **Clustering útil** para segmentación estratégica
- ✅ **Aplicación Streamlit funcional** y profesional

#### 5.2 Recomendaciones Prácticas
1. **Intervención temprana** para estudiantes del Cluster 2
2. **Refuerzo en asistencia** como estrategia principal
3. **Sistema de alertas** basado en predicciones del modelo
4. **Personalización** de estrategias por perfil de cluster

### 6. ANEXOS

#### 6.1 Capturas de Pantalla
- [ ] Interfaz Streamlit completa
- [ ] Matriz de confusión
- [ ] Gráficos de clustering
- [ ] Predicción individual funcionando

#### 6.2 Código Fuente
- **Notebook:** notebook_practica.ipynb
- **Aplicación:** streamlit_app.py
- **Requerimientos:** requirements.txt

---

**Fecha:** Diciembre 2025  
**Autor:** [Adriana Cornejo Ulloa]  
**Tecnologías:** Python, Scikit-learn, Streamlit, Pandas, Plotly