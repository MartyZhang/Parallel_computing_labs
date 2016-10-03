/* Example of using lodepng to load, process, save image */
#include "lodepng.h"
#include <stdio.h>
#include <stdlib.h>


int rectify(int c) {
  if(c >= 127) {
    return c;
  } else {
    return 127;
  }

}

int pickLargest(int j, int k, int l, int m) {
  int largest = 0;
  if(j > largest) {
    largest = j;
  }

  if(k > largest) {
    largest = k;
  }

  if(l > largest) {
    largest = l;
  }

  if(m > largest) {
    largest = m;
  }
  return largest;
}

void process(char* input_filename, char* output_filename)
{
  unsigned error;
  unsigned char *image, *new_image;
  unsigned width, height;

  error = lodepng_decode32_file(&image, &width, &height, input_filename);
  if(error) printf("error %u: %s\n", error, lodepng_error_text(error));
  new_image = malloc(width * height * 4 * sizeof(unsigned char));

  // process image
  unsigned char value;
  for (int i = 0; i < height; i+=2) {
    for (int j = 0; j < width; j+=2) {	
      char square_1_r = image[4*width*i + 4*j + 0];
      char square_1_b = image[4*width*i + 4*j + 0];
      char square_1_g = image[4*width*i + 4*j + 0];

      char square_2_r = image[4*width*i + 4*(j+1) + 0];
      char square_2_g = image[4*width*i + 4*(j+1) + 0];
      char square_2_b = image[4*width*i + 4*(j+1) + 0];

      char square_3_r = image[4*width*(i+1) + 4*j + 0];
      char square_3_g = image[4*width*(i+1) + 4*j + 0];
      char square_3_b = image[4*width*(i+1) + 4*j + 0];

      char square_4_r = image[4*width*(i+1) + 4*(j+1) + 0];
      char square_4_g = image[4*width*(i+1) + 4*(j+1) + 0];
      char square_4_b = image[4*width*(i+1) + 4*(j+1) + 0]; 

      char largest_r = pickLargest(square_1_r, square_2_r, square_3_r, square_4_r); 
      char largest_g = pickLargest(square_1_g, square_2_g, square_3_g, square_4_g); 
      char largest_b = pickLargest(square_1_b, square_2_b, square_3_b, square_4_b); 

      new_image[4*width*i + 4*j + 0] = largest_r; //red
      new_image[4*width*i + 4*j + 1] = largest_g; //green
      new_image[4*width*i + 4*j + 2] = largest_b; //blue

      new_image[4*width*i + 4*(j+1) + 0] = largest_r; //red
      new_image[4*width*i + 4*(j+1) + 1] = largest_g; //green
      new_image[4*width*i + 4*(j+1) + 2] = largest_b; //blue

      new_image[4*width*(i+1) + 4*j + 0] = largest_r; //red
      new_image[4*width*(i+1) + 4*j + 1] = largest_g; //green
      new_image[4*width*(i+1) + 4*j + 2] = largest_b; //blue

      new_image[4*width*(i+1) + 4*(j+1) + 0] = largest_r; //red
      new_image[4*width*(i+1) + 4*(j+1) + 1] = largest_g; //green
      new_image[4*width*(i+1) + 4*(j+1) + 2] = largest_b; //blue
    }
  }

  lodepng_encode32_file(output_filename, new_image, width, height);

  free(image);
  free(new_image);
}

int main(int argc, char *argv[])
{
  char* input_filename = argv[1];
  char* output_filename = argv[2];

  process(input_filename, output_filename);

  return 0;
}
