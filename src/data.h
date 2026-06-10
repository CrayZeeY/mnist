#ifndef DATA_H
#define DATA_H

typedef struct {
    double **images;   /* images[i][pixel] */
    int    *labels;    /* labels[i] = 0..9 */
    int     count;
} Dataset;

Dataset load_mnist(const char *image_file, const char *label_file);
void    free_dataset(Dataset *d);
void    print_image(double *image);

#endif