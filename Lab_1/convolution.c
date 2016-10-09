/* Example of using lodepng to load, process, save image */
#include "lodepng.h"
#include <stdio.h>
#include <stdlib.h>
#include "wm.h"
#include "helpers.h"

int getIndex(int row, int col, int color) {
    return row + col + color;
}


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
    int RED = 0;
    int GREEN = 1;
    int BLUE = 2;
    int ALPHA = 3;

    int r = 0, g = 0, b = 0, a = 0, index = 0;
    for (int i = 1; i < height; i++) {
        for (int j = 1; j < width; j++) {

            for (int ii = 0; ii < weight_size; ii++) {
                for (int jj = 0; jj < weight_size; jj++) {

                    index = (int) (4 * width * (i + ii - 1) + 4 * (j + jj - 1));
                    r += image[index + RED] * w[ii][jj];
                    g += image[index + GREEN] * w[ii][jj];
                    b += image[index + BLUE] * w[ii][jj];
                    //a += image[index + ALPHA] * w[ii][jj];
                }
            }

            new_image[4 * new_width * i + 4 * j + 0] = (unsigned char) clamp(r);
            new_image[4 * new_width * i + 4 * j + 1] = (unsigned char) clamp(g);
            new_image[4 * new_width * i + 4 * j + 2] = (unsigned char) clamp(b);
            new_image[4 * new_width * i + 4 * j + 3] = 255;

            r = g = b = a = 0;
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
