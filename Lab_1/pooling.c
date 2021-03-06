/* Example of using lodepng to load, process, save image */
#include "lodepng.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "helpers.h"

void process(char *input_filename, char *output_filename, int num_threads) {
    unsigned error;
    unsigned char *image, *new_image;
    unsigned width, height;

    error = lodepng_decode32_file(&image, &width, &height, input_filename);
    if (error) printf("error %u: %s\n", error, lodepng_error_text(error));
    unsigned new_width = width / 2;
    unsigned new_height = height / 2;
    new_image = malloc(new_width * new_height * 4 * sizeof(unsigned char));

    clock_t begin = clock();
    int RED = 0, GREEN = 1, BLUE = 2, ALPHA = 3;

    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < height; i += 2) {
        for (int j = 0; j < width; j += 2) {
            int index = 2 * new_width * i + 2 * j;
            new_image[index + RED] = pickLargest(image[4 * width * i + 4 * j],
                                                 image[4 * width * i + 4 * (j + 1)],
                                                 image[4 * width * (i + 1) + 4 * j],
                                                 image[4 * width * (i + 1) + 4 * (j + 1)]);
            new_image[index + GREEN] = pickLargest(image[4 * width * i + 4 * j + 1],
                                                   image[4 * width * i + 4 * (j + 1) + 1],
                                                   image[4 * width * (i + 1) + 4 * j + 1],
                                                   image[4 * width * (i + 1) + 4 * (j + 1) + 1]);
            new_image[index + BLUE] = pickLargest(image[4 * width * i + 4 * j + 2],
                                                  image[4 * width * i + 4 * (j + 1) + 2],
                                                  image[4 * width * (i + 1) + 4 * j + 2],
                                                  image[4 * width * (i + 1) + 4 * (j + 1) + 2]);
            new_image[index + ALPHA] = 255;
        }
    }

    clock_t end = clock();
    float total = 1000.0 * (end - begin) / CLOCKS_PER_SEC;

    printf("runtime is %f ms\n", total);

    lodepng_encode32_file(output_filename, new_image, new_width, new_height);

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
