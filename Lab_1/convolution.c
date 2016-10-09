/* Example of using lodepng to load, process, save image */
#include "lodepng.h"
#include <stdio.h>
#include <stdlib.h>
#include "wm.h"

int getIndex(int row, int col, int color) {
    return row + col + color;
}

void process(char *input_filename, char *output_filename) {
    unsigned error;
    unsigned char *image, *new_image;
    size_t width, height;

    error = lodepng_decode32_file(&image, &width, &height, input_filename);

    if (error) printf("error %u: %s\n", error, lodepng_error_text(error));
    unsigned new_width = width;
    unsigned new_height = height;
    int weight_size = 3;

    new_image = malloc(new_width * new_height * 4 * sizeof(unsigned char));
    int row;
    int col;
    int RED = 0;
    int GREEN = 1;
    int BLUE = 2;
    int ALPHA = 3;

    for (int i = 1; i < height - 1; i+=2) {
        for (int j = 1; j < width - 1; j+=2) {

            for (int ii = 0; ii < weight_size - 1; ii++) {
                for (int jj = 0; jj < weight_size - 1; jj++) {
                    new_image[2 * new_width * i + 2 * j + 0] +=
                            (unsigned char) (image[4 * width * (i + ii - 1) + 4 * (j + jj - 1) + 0] * w[ii][jj]);
                    new_image[2 * new_width * i + 2 * j + 1] +=
                            (unsigned char) (image[4 * width * (i + ii - 1) + 4 * (j + jj - 1) + 1] * w[ii][jj]);
                    new_image[2 * new_width * i + 2 * j + 2] +=
                            (unsigned char) (image[4 * width * (i + ii - 1) + 4 * (j + jj - 1) + 2] * w[ii][jj]);
                }
            }

            new_image[2 * new_width * i + 2 * j + 3] = 255; //Opacity
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
