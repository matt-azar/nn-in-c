#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include <stdio.h>

// Function prototypes
int read_int(FILE *file);
unsigned char **load_images(const char *filename, int *num_images);
unsigned char *load_labels(const char *filename, int *num_labels);
void display_image(unsigned char *image);

#endif // MNIST_LOADER_H
