# Tarea 1c - Modelos de Clasificación Predictiva del Éxito Estudiantil

Trabajo correspondiente a la **Tarea 1c** del curso **Minería y Aprendizaje de Datos (MAD)**, perteneciente al programa de **Magíster en Informática** de la **Universidad Austral de Chile**.

## Objetivo

Construir y evaluar **modelos de clasificación predictiva** para identificar a los estudiantes que completaron el curso (`take_exam = 1` como clase positiva). Se comparan tres familias de modelos:

1. **Regresión Logística** — modelo lineal interpretable.
2. **SVM** — Support Vector Machine con múltiples kernels (linear, rbf, poly).
3. **Random Forest** — propuesta creativa (modelo avanzado).

Todos los modelos se evalúan bajo **validación cruzada estratificada de 5 folds** (`StratifiedKFold`, `random_state=42`), garantizando comparabilidad directa.

## Dataset

Se utiliza el dataset preparado `data_prep.csv` (685 filas × 22 columnas) construido en la Tarea 1a. Las variables motivacionales (`Fi`, `CBi`, `Vi`, `MApi`, `PApi`) contienen strings vacíos y espacios que se limpian con `pd.to_numeric(errors='coerce')`. Tras filtrar filas con datos completos, el dataset final cuenta con **630 filas**.

### Distribución de clases (variable objetivo `take_exam`)

| Clase | Casos | Proporción |
|-------|-------|------------|
| `take_exam = 1` (completó) | 528 | 83.8% |
| `take_exam = 0` (no completó) | 102 | 16.2% |

El dataset está desbalanceado, lo que se mitiga con `class_weight='balanced'` y estratificación en la validación cruzada.

## Selección de variables (Punto 1)

### Variables de uso del sistema

Se calculó la matriz de correlación de Pearson entre las métricas de uso y se detectó colinealidad con umbral |r| > 0.7:

| Par de variables | r | Decisión |
|------------------|---|----------|
| total_time ↔ total_activity | +0.877 | Descartar total_activity |
| total_time ↔ qp_activity | +0.713 | Descartar total_time |
| total_time ↔ total_activity_be | +0.877 | Descartar total_activity_be |
| total_activity ↔ qp_activity | +0.741 | Descartar total_activity |
| total_activity ↔ total_activity_be | +1.000 | Descartar total_activity |
| qp_activity ↔ total_activity_be | +0.741 | Descartar total_activity_be |

**Variable retenida:** `qp_activity` (interacción cognitivamente activa con ejercicios QuizPet/Parsons).

### Variable motivacional

| Variable | Constructo | r con take_exam |
|----------|-----------|-----------------|
| **PApi** | **Performance-Approach** | **+0.0896** |
| CBi | Confianza/Belief | +0.0240 |
| MApi | Mastery-Approach | +0.0192 |
| Fi | Interés | -0.0086 |
| Vi | Valor | -0.0579 |

**Variable seleccionada:** `PApi` (mayor correlación con el objetivo de clasificación).

### Variables finales

| Variable | Tipo | Criterio |
|----------|------|----------|
| `pretest` | Conocimiento previo | Juicio experto (predictor basal obligatorio) |
| `PApi` | Motivacional | Mayor correlación con take_exam |
| `qp_activity` | Engagement | Colinealidad + juicio experto |

## Modelos y resultados

### Regresión Logística (Celda 3)

Modelo lineal con `class_weight='balanced'` y escalado estándar.

- **Accuracy promedio (5-fold):** 0.5206 ± 0.0665
- **Mejor fold (#4):** accuracy = 0.6111, AUC = 0.756, sensibilidad = 0.571, especificidad = 0.810

### SVM (Celda 4)

Se evaluaron tres kernels bajo el mismo esquema de 5-folds:

| Kernel | Accuracy promedio | Std |
|--------|-------------------|-----|
| linear | 0.4000 | 0.0378 |
| rbf | 0.5524 | 0.0599 |
| **poly** | **0.7444** | **0.0535** |

**Mejor kernel:** `poly` — accuracy = 0.7444, AUC = 0.682, sensibilidad = 0.886, especificidad = 0.429

### Random Forest (Celda 5 — propuesta creativa)

Modelo avanzado con 300 árboles, `class_weight='balanced'`. Captura no-linealidades e interacciones sin requerir escalado estricto.

- **Accuracy promedio (5-fold):** 0.6921 ± 0.0464
- **Mejor fold:** AUC = 0.725, sensibilidad = 0.867, especificidad = 0.286

**Importancia de variables (Random Forest):**

| Variable | Importancia |
|----------|-------------|
| qp_activity | 0.440 |
| PApi | 0.343 |
| pretest | 0.217 |

### Tabla resumen comparativa

| Modelo | Accuracy promedio | Std | AUC | Sensibilidad | Especificidad |
|--------|-------------------|-----|-----|--------------|---------------|
| Regresión Logística | 0.5206 | 0.0665 | 0.756 | 0.571 | 0.810 |
| SVM (poly) | 0.7444 | 0.0535 | 0.682 | 0.886 | 0.429 |
| Random Forest | 0.6921 | 0.0464 | 0.725 | 0.867 | 0.286 |

## Análisis

- **Sensibilidad (recall clase positiva):** SVM (poly) y Random Forest detectan mejor a los estudiantes que completan el curso (~87-89%).
- **Especificidad (recall clase negativa):** La Regresión Logística destaca en identificar correctamente a quienes no completan (81%), útil para focalizar intervenciones de retención.
- **AUC:** La Regresión Logística tiene el mayor AUC (0.756), indicando mejor capacidad discriminativa global.
- **Generalización:** Random Forest muestra la menor varianza entre folds (std = 0.046), sugiriendo mejor estabilidad.

## Estructura del repositorio

```
Tarea 1c/
├── README.md
├── Tarea 1c.docx                       # Enunciado de la tarea
├── tara_1c_carlos_ojeda.ipynb          # Notebook con el desarrollo completo
├── Informe_carlos_ojeda_tarea_1c.docx  # Informe final
└── Data/
    ├── student_data.csv                # Datos crudos de estudiantes
    ├── attempts.csv                    # Datos crudos de intentos
    ├── data_prep.csv                   # Dataset preparado (Tarea 1a)
    └── figuras/
        ├── correlacion_uso.png         # Heatmap correlación métricas de uso
        ├── logreg_roc_cm.png           # ROC + matriz de confusión (LogReg)
        ├── svm_roc_cm.png              # ROC + matriz de confusión (SVM)
        ├── rf_roc_cm.png               # ROC + matriz de confusión (Random Forest)
        └── comparacion_roc.png         # Curvas ROC comparativas de los 3 modelos
```

## Requisitos

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Autor

**Carlos Ojeda**
Magíster en Informática - Universidad Austral de Chile
Curso: Minería y Aprendizaje de Datos (MAD)