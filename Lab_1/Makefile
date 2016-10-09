CC=gcc -Wall -std=c11

default:
	$(CC) lodepng_test.c lodepng.c -o rectify
	$(CC) lodepng_test.c pooling.c -o pool
	$(CC) lodepng_test.c lodepng.c -o convolve

rect:
	$(CC) lodepng_test.c lodepng.c -o rectify
	./rectify data/test.png output/rectify.png

pool:
	$(CC) lodepng.c lodepng.h pooling.c -o pool
	./pool data/test.png output/pool.png

conv:
	$(CC) lodepng.c lodepng.h wm.h convolution.c -o convolve
	#./convolve data/test.png output/convolve.png

clean:
	-rm -f rectify
	-rm -f pool
	-rm -f convolve
