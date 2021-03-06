/* Example of using lodepng to load, process, save image */
#include "lodepng.h"
#include <stdio.h>
#include <stdlib.h>
#include "helpers.h"
#include <omp.h>
#include <time.h>

void process(char *input_filename, char *output_filename, int num_threads) {
    unsigned error;
    unsigned char *image, *new_image;
    unsigned width, height;

    error = lodepng_decode32_file(&image, &width, &height, input_filename);
    if (error) printf("error %u: %s\n", error, lodepng_error_text(error));
    new_image = malloc(width * height * 4 * sizeof(unsigned char));
    
    clock_t begin = clock();

    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            new_image[4 * width * i + 4 * j] = rectify(image[4*width*i + 4*j]);
            new_image[4 * width * i + 4 * j + 1] = rectify(image[4*width*i + 4*j + 1]);
            new_image[4 * width * i + 4 * j + 2] = rectify(image[4*width*i + 4*j + 2]);
            new_image[4 * width * i + 4 * j + 3] = 255; //Opacity
        }
    }

    clock_t end = clock();
    double total = (double)(end - begin)/CLOCKS_PER_SEC * 1000.0;

    printf("runtime is %f ms \n", total);
    lodepng_encode32_file(output_filename, new_image, width, height);

    free(image);
    free(new_image);
}

int main(int argc, char *argv[]) {
    char *input_filename = argv[1];
    char *output_filename = argv[2];
    int num_threads = atoi(argv[3]);

    process(input_filename, output_filename, num_threads);

    return 0;
}
