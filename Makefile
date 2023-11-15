all: pgm.o hough_constante hough_compartida hough 

hough:	hough.cu pgm.o
	nvcc hough.cu pgm.o -o hough -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -I/usr/include/opencv4 -w

hough_compartida: hough_compartida.cu pgm.o
	nvcc hough_compartida.cu pgm.o -o hough_compartida -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -I/usr/include/opencv4 -w

hough_constante: hough_constante.cu pgm.o
	nvcc hough_constante.cu pgm.o -o hough_constante -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -I/usr/include/opencv4 -w

pgm.o:	common/pgm.cpp
	g++ common/pgm.cpp -o pgm.o -lopencv_core -lopencv_imgproc -lopencv_highgui -w