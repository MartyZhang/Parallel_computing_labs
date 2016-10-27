/* Example of using lodepng to load, process, save image */
#include "lodepng.h"
#include <stdio.h>
#include <stdlib.h>
#include "wm.h"
#include <omp.h>
#include "helpers.h"
#include <time.h>
#include <pthread.h>

typedef struct {
    int tid;
    int width;
    int start_height, end_height;
    unsigned char *image,*new_image;
} thread_arg_t;

void convolute(void *arg) {
    thread_arg_t *thread_arg = (thread_arg_t *) arg;
    int tid = thread_arg -> tid;
    unsigned char *image = thread_arg -> image;
    unsigned char *new_image = thread_arg -> new_image;
    int width = thread_arg -> width;
    int new_width = width - 2;
    int start_height = thread_arg -> start_height;
    int end_height = thread_arg -> end_height;
    int RED = 0, GREEN = 1, BLUE = 2, ALPHA = 3;
    for (int i = start_height; i < end_height; i++) {
        for (int j = 1; j < width - 1; j++) {
            int r = 0, g = 0, b = 0, index = 0;
            for (int ii = 0; ii < 3; ii++) {
                for (int jj = 0; jj < 3; jj++) {
                    index = (int) (4 * width * (i + ii - 1) + 4 * (j + jj - 1));
                    r += image[index + RED] * w[ii][jj];
                    g += image[index + GREEN] * w[ii][jj];
                    b += image[index + BLUE] * w[ii][jj];
                }
            }

            new_image[4 * new_width * (i-1) + 4 * (j-1) + RED] = clamp(r);
            new_image[4 * new_width * (i-1) + 4 * (j-1) + GREEN] = clamp(g);
            new_image[4 * new_width * (i-1) + 4 * (j-1) + BLUE] = clamp(b);
            new_image[4 * new_width * (i-1) + 4 * (j-1) + ALPHA] = 255;
        }
    }
    pthread_exit(NULL);

}

void process(char *input_filename, char *output_filename, int NUM_THREADS) {
    unsigned error;
    unsigned char *image, *new_image;
    int width, height;

    error = lodepng_decode32_file(&image, &width, &height, input_filename);

    if (error) printf("error %u: %s\n", error, lodepng_error_text(error));
    unsigned new_width = width - 2;
    unsigned new_height = height - 2;

    new_image = malloc(new_width * new_height * 4 * sizeof(unsigned char));

    pthread_t *thread_ids = malloc(NUM_THREADS * sizeof(pthread_t));
    thread_arg_t *thread_args = malloc(NUM_THREADS * sizeof(thread_arg_t));

    clock_t begin = clock();
    
    int chunk_size = height/NUM_THREADS;
    int remainder = height%NUM_THREADS;
    // create threads
    for (int i = 0; i < NUM_THREADS; i++) {
        thread_args[i].tid = i;
        thread_args[i].width = width;
        //edge case at start
        if(i==0) {
            thread_args[i].start_height = 1;
        } else {
            thread_args[i].start_height = i*chunk_size;
        }

        //edge case at end
        if(i == NUM_THREADS - 1) {
            thread_args[i].end_height = (i+1)*chunk_size + remainder - 1;
        } else {
            thread_args[i].end_height = (i+1)*chunk_size;
        }

        thread_args[i].image = image;
        thread_args[i].new_image = new_image;
        pthread_create(&thread_ids[i], NULL, convolute, (void *)&thread_args[i]);
    }

    // join threads
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(thread_ids[i], NULL);
    }


    clock_t end = clock();
    double total = (double)(end - begin)/CLOCKS_PER_SEC;

    printf("runtime is %f \n", total);

    lodepng_encode32_file(output_filename, new_image, new_width, new_height);

    free(image);
    free(new_image);
}
int main(int argc, char *argv[]) {
    char *input_filename = argv[1];
    char *output_filename = argv[2];
    int NUM_THREADS = atoi(argv[3]);
    process(input_filename, output_filename, NUM_THREADS);

    return 0;
}
