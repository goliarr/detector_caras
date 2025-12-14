# Interactive Gesture Meme Filter

Link directo Streamlit: https://detectoracaras.streamlit.app/

Un sistema de visión artificial en tiempo real que detecta expresiones faciales y gestos manuales para superponer memes dinámicamente en la pantalla. Desarrollado con **Python**, **OpenCV** y **MediaPipe**.

EJECUTAR local.py si estás en pc y main.py para web/móvil(también serviría en pc pero va un poco peor)

## Funcionalidades
El sistema utiliza Face Mesh y detección de manos para identificar:
* **Sorpresa**: Detecta la apertura de la boca para mostrar un meme de shock.
* **Felicidad**: Detecta la sonrisa mediante la curvatura de los labios.
* **Giros de cabeza**: Calcula el ratio de la nariz respecto a los bordes de la cara.
* **Mirar arriba**: Calcula la geometría vertical de la cara.
* **Modo Mono**: Detecta si levantas el dedo índice (ignora otros gestos faciales).

## Tecnologías
* **Python 3.x**
* **OpenCV**: Procesamiento de imagen y dibujo en tiempo real.
* **MediaPipe**: Modelos de ML para Face Mesh y Hands.
* **NumPy**: Operaciones matemáticas y manipulación de matrices.

## 📦 Instalación y Uso

1.  Clona el repositorio:
    ```bash
    git clone [https://github.com/goliarr/detector_caras.git](https://github.com/goliarr/detector_caras.git)
    ```
2.  Instala las dependencias:
    ```bash
    pip install -r requirements.txt
    ```
3.  Ejecuta el programa:
    ```bash
    python main.py
    ```

## Cómo funciona (Lógica Matemática)
El proyecto no usa IA genérica, sino trigonometría y geometría sobre landmarks:
* **Ángulos de rotación**: Se calculan comparando distancias relativas entre la nariz y los pómulos.
* **Gestos manuales**: Se evalúa la posición vertical de la punta de los dedos respecto a sus nudillos.

---
Hecho por Gerard
