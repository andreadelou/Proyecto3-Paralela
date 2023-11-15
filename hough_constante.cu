/*
 ============================================================================
 Author        : G. Barlas
 Modified by   : Isabel Solano, Christopher García, Andrea Lam
 Version       : 2.0
 Last modified : November 2023
 License       : Released under the GNU GPL 3.0
 Description   : Constan hough version
 To build use  : make
 ============================================================================
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <string.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "common/pgm.h"

const int degreeInc = 2;
const int degreeBins = 180 / degreeInc;
const int rBins = 100;
const float radInc = degreeInc * M_PI / 180;

//*****************************************************************
// The CPU function returns a pointer to the accummulator
void CPU_HoughTran (unsigned char *pic, int w, int h, int **acc)
{
  float rMax = sqrt (1.0 * w * w + 1.0 * h * h) / 2;  //(w^2 + h^2)/2, radio max equivalente a centro -> esquina
  *acc = new int[rBins * degreeBins];            //el acumulador, conteo depixeles encontrados, 90*180/degInc = 9000
  memset (*acc, 0, sizeof (int) * rBins * degreeBins); //init en ceros
  int xCent = w / 2;
  int yCent = h / 2;
  float rScale = 2 * rMax / rBins;

  for (int i = 0; i < w; i++) //por cada pixel
    for (int j = 0; j < h; j++) //...
      {
        int idx = j * w + i;
        if (pic[idx] > 0) //si pasa thresh, entonces lo marca
          {
            int xCoord = i - xCent;
            int yCoord = yCent - j;  // y-coord has to be reversed
            float theta = 0;         // actual angle
            for (int tIdx = 0; tIdx < degreeBins; tIdx++) //add 1 to all lines in that pixel
              {
                float r = xCoord * cos (theta) + yCoord * sin (theta);
                int rIdx = (r + rMax) / rScale;
                (*acc)[rIdx * degreeBins + tIdx]++; //+1 para este radio r y este theta
                theta += radInc;
              }
          }
      }
}

/** Memoria constante*/
// Almacenamos todos los posibles valores de senos y cosenos
// los inicializamos en el main para pasarlos al device
__constant__ float d_Cos[degreeBins];
__constant__ float d_Sin[degreeBins];

// GPU kernel. One thread per image pixel is spawned.
// The accummulator memory needs to be allocated by the host in global memory
__global__ void GPU_HoughTran(unsigned char *pic, int w, int h, int *acc, float rMax, float rScale) {
    // ya no es necesrio pasar d_Cos ni d_Sin porque estas referencias son globales
    
    int blockID = blockIdx.x;
    int threadID = threadIdx.x;
    int gloID = blockID * blockDim.x + threadID;

    if (gloID >= w * h) return; // in case of extra threads in block

    int xCent = w / 2;
    int yCent = h / 2;

    int xCoord = gloID % w - xCent;
    int yCoord = yCent - gloID / w;

    if (pic[gloID] > 0) {
        for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
            //float r = xCoord * cos(tIdx) + yCoord * sin(tIdx); //probar con esto para ver diferencia en tiempo
            // llamada a d_Cos y d_Sin globales
            float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
            int rIdx = (r + rMax) / rScale;
            //debemos usar atomic, pero que race condition hay si somos un thread por pixel? explique
            atomicAdd(&acc[rIdx * degreeBins + tIdx], 1);
        }
    }
}

// Función para dibujar las líneas más pesadas en la imagen
void drawMostProminentLines(cv::Mat& image, int *h_hough, int w, int h, int rBins, int degreeBins, float radInc) {
    // Busca la línea con el peso más grande en h_hough
    int maxWeight = -1;
    int maxRIdx = -1;
    int maxTIdx = -1;

    for (int rIdx = 0; rIdx < rBins; rIdx++) {
        for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
            int index = rIdx * degreeBins + tIdx;
            int weight = h_hough[index];
            if (weight > maxWeight) {
                maxWeight = weight;
                maxRIdx = rIdx;
                maxTIdx = tIdx;
            }
        }
    }

    // Calcula los valores de theta y r correspondientes a la línea con el mayor peso
    float maxTheta = maxTIdx * radInc;
    float rMax = sqrt(1.0 * w * w + 1.0 * h * h) / 2;
    float rScale = 2 * rMax / rBins;
    float maxR = (maxRIdx * rScale) - rMax;

    // Pinta la línea en la imagen
    for (int i = 0; i < w; i++) {
        int j = static_cast<int>((maxR - i * cos(maxTheta)) / sin(maxTheta));
        if (j >= 0 && j < h) {
            // Colorea el píxel en la imagen original con un color diferente (por ejemplo, rojo)
            image.at<cv::Vec3b>(j, i) = cv::Vec3b(0, 0, 255); // Rojo brillante en formato BGR
        }
    }
}

// Función para comparar los resultados y registrar discrepancias
bool compareResults(int* gpuResult, int* cpuResult, int size) {
    bool match = true;
    for (int i = 0; i < size; i++) {
        if (gpuResult[i] != cpuResult[i]) {
            match = false;
            printf("Discrepancia en el índice %d: GPU = %d, CPU = %d\n", i, gpuResult[i], cpuResult[i]);
        }
    }
    return match;
}


//*****************************************************************
int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Uso: " << argv[0] << " <nombre_de_imagen.pgm>" << std::endl;
        return -1;
    }

    // Load the image using OpenCV
    cv::Mat originalImage = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);

    if (originalImage.empty()) {
        std::cerr << "Error al cargar la imagen." << std::endl;
        return -1;
    }

    int w = originalImage.cols;
    int h = originalImage.rows;

    // reemplazadas por las __constant__
    // float *d_Cos;
    // float *d_Sin;

    // CPU calculation
    int *cpuResult;
    CPU_HoughTran(originalImage.data, originalImage.cols, originalImage.rows, &cpuResult);

    cudaMalloc((void **)&d_Cos, sizeof(float) * degreeBins);
    cudaMalloc((void **)&d_Sin, sizeof(float) * degreeBins);

    // pre-compute values to be stored
    float *pcCos = (float *)malloc(sizeof(float) * degreeBins);
    float *pcSin = (float *)malloc(sizeof(float) * degreeBins);
    float rad = 0;
    for (int i = 0; i < degreeBins; i++) {
        pcCos[i] = cos(rad);
        pcSin[i] = sin(rad);
        rad += radInc;
    }

    float rMax = sqrt(1.0 * w * w + 1.0 * h * h) / 2;
    float rScale = 2 * rMax / rBins;

    cudaMemcpyToSymbol(d_Cos, pcCos, sizeof(float) * degreeBins, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_Sin, pcSin, sizeof(float) * degreeBins, cudaMemcpyHostToDevice);

    // setup and copy data from host to device
    unsigned char *d_in, *h_in;
    int *d_hough, *h_hough;

    h_in = originalImage.data; // h_in contiene los pixeles de la imagen
    h_hough = (int *)malloc(degreeBins * rBins * sizeof(int));

    cudaMalloc((void **)&d_in, sizeof(unsigned char) * w * h);
    cudaMalloc((void **)&d_hough, sizeof(int) * degreeBins * rBins);
    cudaMemcpy(d_in, h_in, sizeof(unsigned char) * w * h, cudaMemcpyHostToDevice);
    cudaMemset(d_hough, 0, sizeof(int) * degreeBins * rBins);
    
    // Marcar el inicio del tiempo de ejecución del kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // execution configuration uses a 1-D grid of 1-D blocks, each made of 256 threads
    //1 thread por pixel
    int blockNum = ceil(w * h / 256);
    GPU_HoughTran<<<blockNum, 256>>>(d_in, w, h, d_hough, rMax, rScale, d_Cos, d_Sin);

    // get results from device
    cudaMemcpy(h_hough, d_hough, sizeof(int) * degreeBins * rBins, cudaMemcpyDeviceToHost);

    // compare CPU and GPU results
    bool resultsMatch = compareResults(h_hough, cpuResult, degreeBins * rBins);

    if (resultsMatch) {
        printf("Los resultados coinciden entre GPU y CPU.\n");
    } else {
        printf("Los resultados difieren entre GPU y CPU.\n");
    }

    // Crea una copia de la imagen original utilizando OpenCV
    cv::Mat imageWithLines;
    cv::cvtColor(originalImage, imageWithLines, cv::COLOR_GRAY2BGR); // Convierte a imagen en color

    // Llama a la función para dibujar las líneas más pesadas
    drawMostProminentLines(imageWithLines, h_hough, w, h, rBins, degreeBins, radInc);

    // Guarda la imagen con las líneas coloreadas utilizando OpenCV
    cv::imwrite("output.png", imageWithLines);

    // Marcar el final del tiempo de ejecución del kernel
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0.0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Tiempo de ejecución del kernel: %f ms\n", milliseconds);

    printf("Done!\n");

    free(pcCos);
    free(pcSin);
    free(h_hough);
    free(cpuResult);
    cudaFree(d_Cos);
    cudaFree(d_Sin);
    cudaFree(d_in);
    cudaFree(d_hough);

    return 0;
}

