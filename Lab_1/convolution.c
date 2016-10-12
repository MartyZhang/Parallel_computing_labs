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
    unsigned int width, height;

    error = lodepng_decode32_file(&image, &width, &height, input_filename);

    if (error) printf("error %u: %s\n", error, lodepng_error_text(error));
    unsigned new_width = width - 2;
    unsigned new_height = height - 2;
    int weight_size = 3;

    new_image = malloc(new_width * new_height * 4 * sizeof(unsigned char));
    int RED = 0, GREEN = 1, BLUE = 2, ALPHA = 3;

    int r = 0, g = 0, b = 0, indexR = 0, indexG = 0, indexB = 0;

    for (int i = 1; i < height; i++) {
        for (int j = 1; j < width; j++) {

            r = g = b = 0;

            for (int ii = 0; ii < weight_size; ii++) {
                for (int jj = 0; jj < weight_size; jj++) {
                    indexR = 4 * width * (i + ii - 1) + 4 * (j + jj - 1);
                    r += image[indexR + RED] * w[ii][jj];
                }
            }

            for (int ii = 0; ii < weight_size; ii++) {
                for (int jj = 0; jj < weight_size; jj++) {
                    indexG = 4 * width * (i + ii - 1) + 4 * (j + jj - 1);
                    g += image[indexG + GREEN] * w[ii][jj];
                }
            }

            for (int ii = 0; ii < weight_size; ii++) {
                for (int jj = 0; jj < weight_size; jj++) {
                    indexB = 4 * width * (i + ii - 1) + 4 * (j + jj - 1);
                    b += image[indexB + BLUE] * w[ii][jj];
                }
            }

            new_image[4 * new_width * (i - 1) + 4 * (j - 1) + RED] = (unsigned char) clamp(r);
            new_image[4 * new_width * (i - 1) + 4 * (j - 1) + GREEN] = (unsigned char) clamp(g);
            new_image[4 * new_width * (i - 1) + 4 * (j - 1) + BLUE] = (unsigned char) clamp(b);
            new_image[4 * new_width * (i - 1) + 4 * (j - 1) + ALPHA] = 255;
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
