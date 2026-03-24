# Regresión de Viviendas: Boston Housing con PyTorch 🏠

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)

Este proyecto utiliza una **Red Neuronal Artificial (ANN)** para resolver un problema de regresión: predecir el valor mediano de las viviendas en Boston basándose en variables sociodemográficas y del entorno.



## 🎯 Descripción del Proyecto

El **Boston Housing Dataset** es un conjunto de datos clásico que contiene información recogida por el Servicio del Censo de los EE.UU. sobre la vivienda en el área de Boston. Consta de 506 registros con **13 variables predictoras** y una variable objetivo (**MEDV**).

### Desafíos Técnicos:
* **Escalado de Datos:** Las variables tienen rangos muy distintos (ej. `CRIM` vs `TAX`). Es crítico normalizar los datos para que el modelo converja correctamente.
* **Naturaleza Continua:** Al ser un problema de regresión, el objetivo no es clasificar, sino minimizar la diferencia numérica entre el precio real y el predicho.

## 🧠 Arquitectura de la Red (MLP)

He implementado un **Perceptrón Multicapa (MLP)** diseñado para alta precisión en regresión:
* **Entrada:** 13 neuronas (una por cada característica: RM, LSTAT, PTRATIO, etc.).
* **Capas Ocultas:** Dos capas densas de **50 neuronas** cada una con funciones de activación **ReLU**.
* **Salida:** 1 neurona con activación **Lineal** para predecir el valor continuo del precio (en miles de dólares).

## 💻 Acceso Rápido al Código

Puedes explorar el proceso de carga, normalización, entrenamiento y las gráficas de pérdida en el cuaderno principal:
  
👉 **[Abrir el Jupyter Notebook: `entrenamiento_boston.ipynb`](/notebooks/entrenamiento_boston.ipynb)**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Alexisfpy/boston-ai-regression-pytorch/blob/master/notebooks/red_neuronal_boston.ipynb)

## 🛠️ Instalación y Configuración

El proyecto está gestionado con `pyproject.toml` y optimizado para detectar automáticamente **GPU (GTX 1050)** o **CPU**.

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/Alexisfpy/boston-ai-regression-pytorch.git
   cd boston-ai-regression-pytorch
2. **Crear y activar el entorno virtual**
   Para este proyecto usaremos -> Python 3.14
   ```bash
   uv python pin 3.14
   ```
   Una vez instalado o si ya tenías Python 3, siguiente comando:
    ```bash
    python -m venv .venv
    ```
    En Windows
    ```bash
    .venv\Scripts\activate
    ```
    En Linux/macOs
    ```bash
    source .venv/bin/activate
    ```
4. **Instalar dependencias**

    Este proyecto utiliza un archivo pyproject.toml para gestionar sus paquetes. Con el entorno virtual activado, instala todas las dependencias automáticamente ejecutando:
    ```bash
    pip install .
    ```
    o
    ```bash
    uv sync
    ```
## 🚀 Cómo ejecutarlo
En Visual Studio Code (Recomendado)
1. Abre la carpeta del proyecto en VS Code.

2. Abre el archivo notebooks/sonar_classification.ipynb.

3. Haz clic en "Select Kernel" (arriba a la derecha) y elige el entorno virtual .venv.

4. Ejecuta las celdas para ver el entrenamiento y los resultados en tiempo real.

### Uso de Modelo entrenado 
Puedes cargar los pesos del modelo (.pth) para realizar inferencia rápida:
```python
import torch
model.load_state_dict(torch.load('modelos/modelo_boston_entrenado.pth'))
model.eval()
```

## 📂 Estructura del Repositorio
```text
├── data/               # Dataset housing.csv
├── modelos/            # Pesos del modelo guardados (.pth)
├── notebooks/          # Notebook interactivo de entrenamiento
├── pyproject.toml      # Configuración de dependencias (uv)
└── README.md           # Documentación del proyecto
```
## 📄 Licencia

Este proyecto está bajo la Licencia MIT - mira el archivo [LICENSE](LICENSE) para más detalles.