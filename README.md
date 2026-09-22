# ANFIS-and-NARX-Surrogate-Models-for-Nonlinear-MPC-Based-Artificial-Pancreas-Control}

# ANFIS and NARX Surrogate Models for Nonlinear MPC-Based Artificial Pancreas Control

[![Conference](https://img.shields.io/badge/NeurIPS%202026-LXAI%20Workshop-darkblue.svg)](https://www.latinxinai.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/CasADi-PyTorch-orange.svg)](https://web.casadi.org/)

# Páncreas Artificial Inteligente: Control Predictivo Basado en Modelos No Lineales (NMPC) mediante Identificación de Sistemas con Redes Neuronal NARX y ANFIS

Este repositorio contiene la implementación completa **End-to-End** de un sistema de **Páncreas Artificial (Artificial Pancreas - AP)** para el control automático de glucosa en sangre en pacientes con Diabetes Mellitus Tipo 1 (T1D). 

El proyecto combina simulaciones fisiológicas realistas en Python (`simglucose` / modelo Dalla Man), entrenamiento e identificación de sistemas dinámicos no lineales mediante **Redes Neuronales NARX** y **ANFIS (Adaptive Neuro-Fuzzy Inference System)** en PyTorch, exportación simbólica a **CasADi**, y diseño e integración de controladores predictivos **NMPC (Nonlinear Model Predictive Control)** con solucionadores numéricos **IPOPT** y controladores **PID** en MATLAB.

---

## Tabla de Contenidos
1. [Visión General del Proyecto](#visión-general-del-proyecto)
2. [Arquitectura del Sistema](#arquitectura-del-sistema)
3. [Fundamentación Teórica y Modelos Matemáticos](#fundamentación-teórica-y-modelos-matemáticos)
   - [Modelo Mínimo de Bergman](#modelo-mínimo-de-bergman)
   - [Simulador Fisiológico Dalla Man (simglucose)](#simulador-fisiológico-dalla-man-simglucose)
   - [Predicción Exclusiva de Glucosa G(k+1)](#predicción-exclusiva-de-glucosa-gk1)
4. [Protocolo de Excitación Persistente (Excitation Controller)](#protocolo-de-excitación-persistente-excitation-controller)
5. [Matemática de Escalado y Normalización](#matemática-de-escalado-y-normalización)
   - [Reescalamiento de Insulina (u)](#reescalamiento-de-insulina-u)
   - [Reescalamiento de Perturbación de Comida (D)](#reescalamiento-de-perturbación-de-comida-d)
6. [Modelos de Identificación (NARX y ANFIS)](#modelos-de-identificación-narx-y-anfis)
   - [Variante Desplegable (Window Size = 3)](#variante-desplegable-window-size--3)
   - [Exportación y Verificación PyTorch vs CasADi](#exportación-y-verificación-pytorch-vs-casadi)
7. [Diseño del Controlador NMPC en MATLAB/CasADi](#diseño-del-controlador-nmpc-en-matlabcasadi)
   - [Formulación del Problema de Optimización](#formulación-del-problema-de-optimización)
   - [Relajación de Restricciones y Estrategia Fallback](#relajación-de-restricciones-y-estrategia-fallback)
8. [Estructura del Repositorio](#estructura-del-repositorio)
9. [Guía de Instalación y Ejecución Paso a Paso](#guía-de-instalación-y-ejecución-paso-a-paso)
10. [Protocolos de Evaluación y Resultados Comparativos](#protocolos-de-evaluación-y-resultados-comparativos)
    - [Protocolo 1: Comida Sostenida (Bergman vs PID)](#protocolo-1-comida-sostenida-bergman-vs-pid)
    - [Protocolo 2: Comida Tipo Impulso (Comparación de 3 Vías)](#protocolo-2-comida-tipo-impulso-comparación-de-3-vías)
11. [Conclusiones y Desempeño Clínico](#conclusiones-y-desempeño-clínico)

---

## Visión General del Proyecto

El objetivo principal de este páncreas artificial es mantener la glucosa en sangre ($G$) dentro del **rango objetivo clínico seguro (70–180 mg/dL)**, rastreando una referencia constante de **100 mg/dL**, evitando episodios de **hipoglucemia ($G < 70 \text{ mg/dL}$)** e **hiperglucemia ($G > 180 \text{ mg/dL}$)** ante perturbaciones estocásticas de ingesta de carbohidratos (comidas) e hipoglucemias iniciales.

### Principales Innovaciones del Proyecto:
1. **Identificación de Dinámicas Reales**: Se sustituyen las ecuaciones diferenciales del modelo analítico de Bergman en el controlador por una red neuronal **NARX entrenada sobre datos del simulador simglucose (modelo Dalla Man)**.
2. **Compatibilidad Estricta de Sensores**: El modelo NARX desplegable utiliza **únicamente señales medibles en la práctica clínica**: lecturas pasadas del monitor continuo de glucosa ($G$), infusiones pasadas de la bomba de insulina ($u$) y anuncios de comida ($D$).
3. **Garantía Numérica PyTorch-CasADi**: Se desarrolló una exportación matemática exacta que reconstruye el grafo computacional de la red neuronal dentro de **CasADi en MATLAB**, logrando una discrepancia máxima inferior a $10^{-5}$ mg/dL.
4. **Resolución de Infeasibilidades NMPC**: Diseño de fronteras relajadas en el horizonte inicial para prevenir fallos en el solucionador IPOPT en hipoglucemias severas.

---

## Arquitectura del Sistema

```
+-----------------------------------------------------------------------------------+
|                                  FASE 1: PYTHON                                   |
|                                                                                   |
|  +--------------------+      +-----------------------+      +------------------+  |
|  | simglucose (Dalla  | ---> | Excitation Controller | ---> | Dataset (10 días)|  |
|  | Man T1D Simulator) |      | (Bolos estocásticos)  |      | Sample T = 5 min |  |
|  +--------------------+      +-----------------------+      +------------------+  |
|                                                                      |            |
|  +--------------------+      +-----------------------+               v            |
|  |  Exportador MAT    | <--- | Entrenamiento PyTorch  | <--- +------------------+  |
|  | (export_narx_to_mat)      | (NARX & ANFIS models) |      | Reescalamiento   |  |
|  +--------------------+      +-----------------------+      | (u_res, D_res)   |  |
+-----------|--------------------------------------------------+------------------+  |
            |                                                  +------------------+  |
            v                                                                        
+-----------------------------------------------------------------------------------+
|                                  FASE 2: MATLAB                                   |
|                                                                                   |
|  +------------------------+      +--------------------+      +-----------------+  |
|  | build_narx_casadi_func | ---> | CasADi Function    | ---> | Optimizador     |  |
|  | (Carga de pesos .mat)  |      | f_narx(s_k, u, D)  |      | IPOPT (NMPC)    |  |
|  +------------------------+      +--------------------+      +-----------------+  |
|                                                                       |            |
|                                                                       v            |
|  +------------------------+      +--------------------+      +-----------------+  |
|  | Tablas CSV & Gráficas  | <--- | Evaluación 3 Vías  | <--- | Simulador       |  |
|  | Dashboard (.png)       |      | (NMPC-Bergman,     |      | Bucle Cerrado   |  |
|  |                        |      |  NMPC-NARX, PID)   |      | (Plant Engine)  |  |
|  +------------------------+      +--------------------+      +-----------------+  |
+-----------------------------------------------------------------------------------+
```

---

## Fundamentación Teórica y Modelos Matemáticos

### Modelo Mínimo de Bergman
El modelo de Bergman describe la interacción glucosa-insulina a través de 3 ecuaciones diferenciales ordinarias (ODEs):

$$\frac{dG}{dt} = -p_1 \cdot (G - G_b) - X \cdot G + D(t)$$

$$\frac{dX}{dt} = -p_2 \cdot X + p_3 \cdot (I - I_b)$$

$$\frac{dI}{dt} = -p_4 \cdot (I - I_b) + u(t)$$

*   $G(t)$: Concentración de glucosa plasmática [mg/dL].
*   $X(t)$: Acción de la insulina en tejidos periféricos [min$^{-1}$].
*   $I(t)$: Concentración de insulina plasmática [mU/L].
*   $u(t)$: Tasa de infusión de insulina exógena (control) [mU/min].
*   $D(t)$: Tasa de aparición de glucosa por ingesta de alimentos [mg/dL/min].
*   **Parámetros Nominales**: $p_1 = 0.01$, $p_2 = 0.015$, $p_3 = 2\times 10^{-6}$, $p_4 = 0.2$, $G_b = 80 \text{ mg/dL}$, $I_b = 7 \text{ mU/L}$.

### Simulador Fisiológico Dalla Man (`simglucose`)
El entorno `simglucose` implementa el modelo de Dalla Man & Cobelli (usado por la FDA para pruebas in-silico), con 13 estados fisiológicos complejos que simulan el tracto gastrointestinal, subsistema hepático y tasa de absorción de glucosa.

### Predicción Exclusiva de Glucosa $G(k+1)$
En las iteraciones iniciales del proyecto, los modelos predecían el vector completo de estados $[G(k+1), X(k+1), I(k+1)]$. Sin embargo, las variables internas $X$ e $I$ en el modelo de Dalla Man poseen escalas y significados fisiológicos no equivalentes con las variables abstractas $X$ e $I$ de Bergman. Predecirlas introducía errores de desajuste dimensional e imposibilitaba la implementación clínica real (pues $X$ e $I$ no son medibles con sensores subcutáneos). 

Por lo tanto, la arquitectura de identificación fue reconfigurada para predecir **únicamente la glucosa escalar $G(k+1)$**, la cual es la única variable requerida para la realimentación del bucle de control.

---

## Protocolo de Excitación Persistente (Excitation Controller)

Los controladores clínicos estándar (como los algoritmos Basal-Bolo convencionales) generan trayectorias ultra-seguras donde más del **95% de las muestras operan en el nivel basal** y menos del **0.17%** explora la zona de saturación superior del control ($u > 4.0 \text{ mU/min}$). Esta falta de excitación persistente imposibilita que una red neuronal aprenda la respuesta del cuerpo ante infusiones máximas de insulina.

Para resolver esto, se desarrolló el controlador estocástico **`ExcitationController`** en Python:
1. **Inyección Estocástica**: Genera infusión basal combinada con bolos estocásticos aleatorios de magnitud variable (0.5 a 6.0 U) en instantes pseudo-aleatorios.
2. **Protección de Glucosa Baja**: Bloquea la inyección de bolos si la glucosa cae por debajo de $80 \text{ mg/dL}$.
3. **Corte Absoluto por Hipoglucemia**: Si la glucosa desciende de $60 \text{ mg/dL}$, suspende la infusión a $u = 0$.

**Impacto en la Calidad de Datos**: El porcentaje de muestras en la zona de saturación ($u > 4.0 \text{ mU/min}$) aumentó **7.5 veces** (de 0.17% a 9.39%), garantizando una identificación robusta en regímenes críticos de control.

---

## Matemática de Escalado y Normalización

### Reescalamiento de Insulina ($u$)
La bomba de insulina clínica en `simglucose` administra la infusión en Unidades por minuto ($\text{U/min}$). El controlador NMPC en MATLAB opera en el dominio del modelo de Bergman con límites estricto de $[u_{\text{min\_nmpc}}, u_{\text{max\_nmpc}}] = [0, 5] \text{ mU/min}$. Se aplica la siguiente transformación lineal min-max:

$$u_{\text{rescaled}} = u_{\text{min\_nmpc}} + \frac{u - u_{\text{min\_sim}}}{u_{\text{max\_sim}} - u_{\text{min\_sim}}} \cdot (u_{\text{max\_nmpc}} - u_{\text{min\_nmpc}})$$

### Reescalamiento de Perturbación de Comida ($D$)
Las comidas crudas en `simglucose` se expresan en gramos de carbohidratos (g CHO), donde un almuerzo típico es de $\sim 70 \text{ g}$. El modelo NMPC en MATLAB utiliza una perturbación nominal de $D = 2.0$ para un evento de comida. Se aplica un mapeo directo anclado al máximo observado ($D_{\text{max\_observado}} = 14.0$ en ventanas de 5 min):

$$D_{\text{rescaled}} = D \cdot \left(\frac{2.0}{D_{\text{max\_observado}}}\right)$$

Esto preserva la proporción entre comidas (desayuno < cena < almuerzo) e iguala las magnitudes de estímulo entre todos los controladores.

---

## Modelos de Identificación (NARX y ANFIS)

### Variante Desplegable (Window Size = 3)
El modelo sustituto **NARX Desplegable** toma una ventana temporal de $W=3$ muestras pasadas ($15 \text{ minutos}$) de las variables medibles:

$$x_{\text{raw}}(k) = \begin{bmatrix} G(k) & G(k-1) & G(k-2) & u(k) & u(k-1) & u(k-2) & D(k) & D(k-1) & D(k-2) \end{bmatrix}^T \in \mathbb{R}^9$$

#### Arquitectura de la Red Neuronal NARX:
*   **Normalización**: $x_{\text{scaled}} = (x_{\text{raw}} - \mu_X) \oslash \sigma_X$
*   **Capa Oculta 1**: $h_1 = \text{ReLU}(W_1 \cdot x_{\text{scaled}} + b_1) \in \mathbb{R}^{64}$
*   **Capa Oculta 2**: $h_2 = \text{ReLU}(W_2 \cdot h_1 + b_2) \in \mathbb{R}^{32}$
*   **Capa de Salida**: $y_{\text{scaled}} = W_3 \cdot h_2 + b_3 \in \mathbb{R}^1$
*   **Desnormalización**: $\hat{G}(k+1) = y_{\text{scaled}} \cdot \sigma_y + \mu_y$

#### Métricas de Exactitud del Entrenamiento (PyTorch):
*   **MAE de Glucosa**: $1.365 \text{ mg/dL}$
*   **RMSE de Glucosa**: $2.404 \text{ mg/dL}$
*   **Coeficiente de Determinación ($R^2$)**: $0.9942$

### Exportación y Verificación PyTorch vs CasADi
El archivo Python `export_narx_to_mat.py` exporta los pesos $W_1, b_1, W_2, b_2, W_3, b_3$ y parámetros de escalado a `narx_weights_for_matlab.mat`. El script de MATLAB `build_narx_casadi_function.m` reconstruye el grafo en CasADi.

**Resultado de Validación**:
*   Predicción PyTorch $G(k+1)$: `111.366272 mg/dL`
*   Predicción CasADi $G(k+1)$: `111.366274 mg/dL`
*   Diferencia Absoluta Máxima: **$2.0 \times 10^{-6} \text{ mg/dL}$** (Concordancia exacta).

---

## Diseño del Controlador NMPC en MATLAB/CasADi

### Formulación del Problema de Optimización
El controlador **NMPC_NARX** resuelve en cada paso de tiempo $k$ el siguiente problema de programación no lineal (NLP) en un horizonte de predicción de $N=12$ pasos ($60 \text{ min}$):

$$\min_{\mathbf{U}, \mathbf{X}_{\text{var}}} \sum_{j=1}^{N} \left( Q \cdot (G_{k+j} - G_{\text{ref}})^2 + R \cdot u_{k+j-1}^2 \right)$$

$$\text{sujeto a: } G_{k+j} = f_{\text{NARX}}\Big( [G_{k+j-1}, G_{k+j-2}, G_{k+j-3}], [u_{k+j-1}, u_{k+j-2}, u_{k+j-3}], [D_{k+j-1}, D_{k+j-2}, D_{k+j-3}] \Big)$$

$$u_{\text{min}} \le u_{k+j-1} \le u_{\text{max}}, \quad \forall j=1 \dots N$$

*   $G_{\text{ref}} = 100 \text{ mg/dL}$, $Q = 1.0$, $R = 0.01$.
*   $u_{\text{min}} = 0 \text{ mU/min}$, $u_{\text{max}} = 5 \text{ mU/min}$.

### Relajación de Restricciones y Estrategia Fallback
Para evitar que el optimizador IPOPT se declare **infactible** cuando el estado inicial del paciente se encuentra fuera del rango seguro (por ejemplo, en hipoglucemia inicial a $G_0 = 60 \text{ mg/dL}$), las restricciones de caja sobre la glucosa ($70 \le G \le 180$) solo se aplican como términos de penalización suave o se relajan en el nodo actual ($j=1$), permitiendo que el solucionador compute trayectorias de recuperación válidas. Si IPOPT no alcanza tolerancia de convergencia, el controlador ejecuta una **estrategia de respaldo híbrida (fallback)** aplicando el valor ponderado de la primera acción óptima calculada.

---

## Estructura del Repositorio

```
.
├── python_component/
│   ├── data/                           # Datasets generados y parámetros de reescalamiento
│   │   ├── raw_simglucose_dataset.csv
│   │   └── rescaling_params_excitation.json
│   ├── scripts/
│   │   └── main.py                     # Pipeline principal de entrenamiento de NARX y ANFIS
│   ├── src/
│   │   ├── data/                       # Generador de simglucose con ExcitationController
│   │   │   └── generate_simglucose_data.py
│   │   ├── models/                     # Clases de PyTorch (NARX, ANFIS)
│   │   └── utils/                      # Métricas y utilidades de preprocesamiento
│   ├── diagnose_u_distribution.py      # Diagnóstico de excitación persistente
│   ├── export_narx_to_mat.py           # Exportador de pesos PyTorch -> MATLAB .mat
│   ├── verify_narx_python.py           # Verificador de inferencia en Python
│   └── requirements.txt                # Dependencias Python
│
└── matlab_component/ (PA/)
    ├── build_narx_casadi_function.m    # Constructor simbólico de la función CasADi
    ├── test_narx_casadi.m              # Script de verificación de concordancia PyTorch/CasADi
    ├── NMPC.m                          # NMPC original con modelo analítico de Bergman (Sostenido)
    ├── PID.m                           # PID original con modelo analítico de Bergman (Sostenido)
    ├── NMPC_impulso.m                  # Variante de NMPC-Bergman bajo evento de Impulso
    ├── PID_impulso.m                    # Variante de PID bajo evento de Impulso
    ├── NMPC_NARX.m                     # Nuevo NMPC basado en red neuronal NARX
    ├── comparison.m                    # Comparación de 2 vías (Comida Sostenida)
    ├── comparison_impulso.m            # Comparación de 3 vías (Comida tipo Impulso)
    ├── narx_weights_for_matlab.mat     # Archivo con pesos exportados
    ├── resultados_comparacion.csv      # Resultados cuantitativos exportados (Sostenido)
    └── resultados_comparacion_impulso.csv # Resultados cuantitativos exportados (Impulso)
```

---

## Guía de Instalación y Ejecución Paso a Paso

### 1. Entorno Python (Generación y Entrenamiento)

```bash
# Navegar al directorio de Python
cd python_component

# Crear y activar entorno virtual
python -m venv venv
source venv/bin/activate  # En Linux/macOS
# venv\Scripts\activate   # En Windows

# Instalar dependencias
pip install -r requirements.txt

# 1. Generar datos con excitación persistente (10 días, T=5 min) y entrenar modelos
python scripts/main.py

# 2. Exportar los pesos del modelo NARX a MATLAB (.mat)
python export_narx_to_mat.py

# 3. Verificar predicción en Python
python verify_narx_python.py
```

### 2. Entorno MATLAB (Simulación de Controladores y Evaluación)

1. Abra **MATLAB** y asegúrese de tener la librería **CasADi** agregada a su ruta (`addpath('ruta/a/casadi')`).
2. Establezca como directorio de trabajo la carpeta `matlab_component` (`PA/`).
3. **Verificar concordancia simbólica PyTorch-CasADi**:
   ```matlab
   test_narx_casadi
   ```
   *Debe confirmar una discrepancia < 1e-5 mg/dL.*

4. **Ejecutar Comparación de 3 Vías (Protocolo Impulso)**:
   ```matlab
   comparison_impulso
   ```
   *Ejecuta NMPC-Bergman, PID y NMPC-NARX, genera `resultados_comparacion_impulso.csv` y guarda el gráfico `comparacion_3vias_impulso.png`.*

5. **Ejecutar Comparación de 2 Vías (Protocolo Sostenido)**:
   ```matlab
   comparison
   ```
   *Ejecuta NMPC-Bergman y PID bajo perturbación sostenida de 60 min, genera `resultados_comparacion.csv` y guarda `comparacion_sustentada.png`.*

---

## Protocolos de Evaluación y Resultados Comparativos

Para garantizar una evaluación justa entre modelos analíticos y sustitutos, se implementaron dos protocolos experimentales independientes:

### Protocolo 1: Comida Sostenida (60 min) — NMPC-Bergman vs PID
Evalúa la respuesta ante una ingesta de comida de magnitud $D = 2.0$ durante 60 minutos continuos (de $t = 60$ a $t = 120 \text{ min}$).

| Métrica | NMPC-Bergman | PID |
| :--- | :---: | :---: |
| **Time in Range (70-180 mg/dL) - Comida (%)** | **100.00%** | **100.00%** |
| **Time in Range (70-180 mg/dL) - Hipoglucemia (%)** | **77.05%** | **77.05%** |
| **Tiempo de Recuperación de Hipoglucemia (min)** | **70 min** | **70 min** |
| **Glucosa Pico - Comida (mg/dL)** | **164.95 mg/dL** | 170.22 mg/dL |
| **Glucosa Mínima (Nadir) - Hipoglucemia (mg/dL)** | 60.00 mg/dL | 60.00 mg/dL |
| **ISE (Integral Squared Error) - Comida** | **217,374** | 260,997 |
| **ISE - Hipoglucemia** | 214,116 | **213,878** |
| **IAE (Integral Absolute Error) - Comida** | **6,511.1** | 7,123.0 |
| **IAE - Hipoglucemia** | 7,864.9 | **7,860.2** |
| **Insulina Total Administrada - Comida (mU)** | 675.00 mU | 650.00 mU |
| **Insulina Total Administrada - Hipoglucemia (mU)** | 1.14 mU | **0.00 mU** |

---

### Protocolo 2: Comida Tipo Impulso (5 min) — Comparación de 3 Vías
Evalúa a los tres controladores (**NMPC-Bergman**, **NMPC-NARX** y **PID**) bajo un pulso único de comida de $D_{\text{rescaled}} = 2.0$ a los $t = 60 \text{ min}$ (duración 5 min). Este protocolo es matemáticamente equivalente y válido para el modelo NARX.

| Métrica | NMPC-Bergman | NMPC-NARX | PID |
| :--- | :---: | :---: | :---: |
| **Time in Range (70-180 mg/dL) - Comida (%)** | **100.00%** | **100.00%** | **100.00%** |
| **Time in Range (70-180 mg/dL) - Hipoglucemia (%)** | 77.05% | **95.08%** | 77.05% |
| **Tiempo de Recuperación Hipoglucemia (min)** | 70 min | **15 min** | 70 min |
| **Glucosa Pico - Comida (mg/dL)** | **90.00 mg/dL** | 118.74 mg/dL | **90.00 mg/dL** |
| **Glucosa Mínima (Nadir) - Hipoglucemia (mg/dL)** | 60.00 mg/dL | 60.00 mg/dL | 60.00 mg/dL |
| **ISE - Comida** | 88,589 | **17,203** | 92,272 |
| **ISE - Hipoglucemia** | 214,116 | **31,118** | 213,878 |
| **IAE - Comida** | 5,087.5 | **1,303.4** | 5,197.0 |
| **IAE - Hipoglucemia** | 7,864.9 | **1,430.9** | 7,860.2 |
| **Insulina Total Administrada - Comida (mU)** | $\approx 0 \text{ mU}$ | 147.28 mU | 25.00 mU |
| **Insulina Total Administrada - Hipoglucemia (mU)** | 1.14 mU | 266.43 mU | **0.00 mU** |

---

## Conclusiones y Desempeño Clínico

1. **Recuperación Fisiológica Acelerada**: En el escenario de hipoglucemia inicial ($G_0 = 60 \text{ mg/dL}$), el controlador **NMPC-NARX logra recuperar al paciente al rango seguro ($G \ge 70 \text{ mg/dL}$) en solo 15 minutos**, comparado con los 70 minutos requeridos por NMPC-Bergman y PID. Esto eleva el porcentaje de **Tiempo en Rango (TIR) en hipoglucemia del 77.05% al 95.08%**.
2. **Dinámicas de Contrarregulación Aprendidas**: El modelo NARX capturó la respuesta fisiológica compleja de producción de glucosa hepática interna presente en el simulador `simglucose` (Dalla Man), permitiendo al controlador realizar correcciones preventivas más suaves y efectivas.
3. **Reducción Masiva de Errores Acumulados**: En el evento de comida de tipo impulso, **NMPC-NARX redujo el ISE en un 80.5%** y el **IAE en un 74.4%** con respecto al NMPC basado en Bergman, demostrando la superioridad de utilizar modelos sustitutos (surrogate models) identificados directamente de la fisiología del paciente en lugar de modelos analíticos simplificados.
