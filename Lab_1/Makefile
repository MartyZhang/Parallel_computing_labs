CC=gcc-5 -fopenmp -Wall -std=c11

default:
	$(CC) lodepng_test.c rectify.c -o rectify
	$(CC) lodepng_test.c pooling.c -o pool
	$(CC) lodepng_test.c lodepng.c -o convolve

rect:
	$(CC) helpers.c lodepng.c rectify.c -o rectify
	./rectify data/test.png output/rectify.png

pool:
	$(CC) helpers.c lodepng.c pooling.c -o pool
	./pool data/test.png output/pool.png

conv:
	$(CC) lodepng.c lodepng.h wm.h convolution.c -o convolve
	#./convolve data/test.png output/convolve.png

clean:
	-rm -f rectify
	-rm -f pool
	-rm -f convolve
