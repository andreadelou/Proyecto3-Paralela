## *Proyecto Final - Computación Paralela y Distribuida*

### Descripción

El proyecto se centra en la implementación en CUDA del algoritmo de la Transformada de Hough, una técnica computacional fundamental para la detección de líneas rectas en imágenes. Esta transformación se utiliza en imágenes en blanco y negro, generalmente después de procesos de detección de bordes como Canny o Sobel. Su aplicación forma parte de los filtros para la detección precisa de bordes en imágenes. 

El propósito principal de este proyecto es realizar una implementación eficiente en CUDA de este algoritmo, permitiendo la detección precisa de líneas en imágenes, lo que puede ser fundamental en aplicaciones de visión por computadora, reconocimiento de patrones y procesamiento de imágenes en general. Se implementaron 3 versiones, un programa base, un programa con memoria compartida y un programa con memoria constante.

### Uso
- Archivos necesarios para compilación y ejecución
    - Makefile
    - Carpeta common
    - runway.pgm (imagen de prueba)
    - hough.cu
    - hough_compartida.cu
    - hough_constante.cu

### Comandos para ejecución
#### Compilación
```
make
```
#### Ejecución
```
./<nombre objeto> runway.pgm
```
- (Nombre de objetos: hough, hough_compartida, hough_constante)

#### Resultado programa base
![output_1](https://github.com/andreadelou/Proyecto3-Paralela/assets/60325784/599ba4b3-7c44-4ef4-8077-a32c399ac564)
