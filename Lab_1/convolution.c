/* Example of using lodepng to load, process, save image */
#include "lodepng.h"
#include <stdio.h>
#include <stdlib.h>
#include "wm.h"
#include <omp.h>
#include "helpers.h"

void process(char *input_filename, char *output_filename) {
    unsigned error;
    unsigned char *image, *new_image;
    int width, height;

    error = lodepng_decode32_file(&image, &width, &height, input_filename);

    if (error) printf("error %u: %s\n", error, lodepng_error_text(error));
    unsigned new_width = width - 2;
    unsigned new_height = height - 2;
    int weight_size = 3;

    new_image = malloc(new_width * new_height * 4 * sizeof(unsigned char));
    int RED = 0, GREEN = 1, BLUE = 2, ALPHA = 3;
#pragma omp parallel for
    for (int i = 1; i < height; i++) {
#pragma omp parallel for
        for (int j = 1; j < width; j++) {

            int r = 0, g = 0, b = 0, a = 0, indexR = 0, indexG = 0, indexB = 0;
            for (int ii = 0; ii < weight_size; ii++) {
#pragma omp parallel for reduction (+:r)
                for (int jj = 0; jj < weight_size; jj++) {
                    indexR = (int) (4 * width * (i + ii - 1) + 4 * (j + jj - 1));
                    r += image[indexR + RED] * w[ii][jj];
                }
            }

            for (int ii = 0; ii < weight_size; ii++) {
#pragma omp parallel for reduction (+:g)
                for (int jj = 0; jj < weight_size; jj++) {
                    indexG = (int) (4 * width * (i + ii - 1) + 4 * (j + jj - 1));
                    g += image[indexG + GREEN] * w[ii][jj];
                }
            }

            for (int ii = 0; ii < weight_size; ii++) {
#pragma omp parallel for reduction (+:b)
                for (int jj = 0; jj < weight_size; jj++) {
                    indexB = (int) (4 * width * (i + ii - 1) + 4 * (j + jj - 1));
                    b += image[indexB + BLUE] * w[ii][jj];
                }
            }

            new_image[4 * new_width * i + 4 * j + RED] = (unsigned char) clamp(r);
            new_image[4 * new_width * i + 4 * j + GREEN] = (unsigned char) clamp(g);
            new_image[4 * new_width * i + 4 * j + BLUE] = (unsigned char) clamp(b);
            new_image[4 * new_width * i + 4 * j + ALPHA] = 255;
        }
    }

    lodepng_encode32_file(output_filename, new_image, new_width, new_height);

    free(image);
    free(new_image);
}

int main(int argc, char *argv[]) {
    char *input_filename = argv[1];
    char *output_filename = argv[2];

    process(input_filename, output_filename);

    return 0;
}
