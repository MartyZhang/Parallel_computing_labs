CC=gcc -fopenmp -Wall -std=c11

default:
	Make rect
	Make pool
	Make conv

rect:
	$(CC) helpers.c lodepng.c rectify.c -o rectify
	./rectify data/test.png output/rectify.png 1

pool:
	$(CC) helpers.c lodepng.c pooling.c -o pool
	./pool data/test.png output/pool.png 1

conv:
	$(CC) helpers.c lodepng.h lodepng.c wm.h convolution_pthread.c -o convolve
	./convolve data/test.png output/convolve.png 1

clean:
	-rm -f rectify
	-rm -f pool
	-rm -f convolve