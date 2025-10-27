# 🩺 Revisión de Pares CBIS-DDSM

Aplicación interactiva desarrollada en **Python** y **Streamlit** para revisar, validar y anotar pares de imágenes del conjunto de datos **CBIS-DDSM** (Curated Breast Imaging Subset of the Digital Database for Screening Mammography).

Permite visualizar pares de mamografías *antes* y *después* de una lesión, superponer máscaras de segmentación y registrar decisiones de revisión (“ACEPTAR” o “CANCELAR”) directamente en un archivo Excel.

---

## 🚀 Características principales

- Interfaz gráfica moderna basada en **Streamlit**.  
- Carga automática de los pares desde `pairs.xlsx`.  
- Superposición de máscaras de segmentación sobre las mamografías.  
- Cálculo automático del porcentaje de crecimiento tumoral (`growth_pct`).  
- Navegación rápida entre pares (`Primero`, `Anterior`, `Siguiente`, `Último`).  
- Botones para **Aceptar**, **Cancelar** o **Limpiar decisiones**.  
- Visualización de progreso general (pendientes, revisados, totales).  
- Tabla expandible con las rutas e información resumida de cada par.

---

## ⚙️ Requisitos

- Python 3.9 o superior  
- Entorno virtual (recomendado)  
- Librerías principales:
  - `streamlit`
  - `pandas`
  - `numpy`
  - `Pillow`
  - `scipy`
  - `openpyxl`

Instalación rápida:

```bash
pip install -r requirements.txt
```

---

## 🧩 Ejecución local

Desde la carpeta raíz del proyecto:

```bash
streamlit run review_pairs.py
```

Luego abre en tu navegador:

```bash
http://localhost:8501
```

---

## 📊 Estructura esperada del archivo `pairs.xlsx`

La hoja `pairs` debe contener al menos las siguientes columnas:

| Columna          | Descripción                                    |
|------------------|------------------------------------------------|
| before_image     | Ruta a la imagen de la mamografía "antes"      |
| after_image      | Ruta a la imagen de la mamografía "después"    |
| before_mask      | Ruta a la máscara asociada (opcional)          |
| after_mask       | Ruta a la máscara asociada (opcional)          |
| size_before_px   | Tamaño del tumor antes                         |
| size_after_px    | Tamaño del tumor después                       |
| pathology        | Tipo de patología (benigna/maligna)            |
| view             | Vista (CC, MLO, etc.)                          |
| review           | Columna de decisión (ACEPTAR / CANCELAR)       |

> Si no existe la columna `review`, el sistema la crea automáticamente.

---

## 💾 Decisiones de revisión

- ✅ **ACEPTAR** — guarda el par como válido.  
- ❌ **CANCELAR** — descarta el par.  
- 🧼 **Limpiar decisión** — elimina la decisión del par actual.  
- 🧹 **Limpiar todos** — borra todas las decisiones de la hoja.  

Todas las modificaciones se guardan directamente en el Excel (`pairs.xlsx`).

---

## 🖼️ Visualización de máscaras

Las imágenes y máscaras se cargan desde las rutas especificadas en el Excel.  
El sistema resalta automáticamente el borde de la lesión usando dilatación binaria (`scipy.ndimage.binary_dilation`) para una visualización clara y rápida.

---

## 👩‍🔬 Contexto del proyecto

Esta aplicación forma parte de un sistema experimental para el **pronóstico de evolución tumoral en mastografías**, utilizando pares de imágenes anotadas por radiólogos.  
La herramienta facilita la validación humana de los pares de entrenamiento utilizados en redes generativas adversariales condicionales (**cGAN**).

---

## 📄 Licencia

Este proyecto se distribuye bajo la licencia **CIC**.  
Puedes usarlo, modificarlo y adaptarlo libremente citando la fuente original.

---

## ✉️ Autor

**Ariel Plasencia Díaz**  
Maestría en Ciencias en Ingeniería de Cómputo  
Laboratorio de Robótica y Mecatrónica
Centro de Investigación en Computación del Instituto Politécnico Nacional  
📧 [aplascenciad2024@cic.ipn.mx](mailto:aplascenciad2024@cic.ipn.mx)
