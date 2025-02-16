#ifndef DATASET_H
#define DATASET_H

typedef struct {
    float *images;    
    int *labels;      
    int num_samples;
} Dataset;

Dataset load_cifar10_training(const char *data_dir);
Dataset load_cifar10_test(const char *data_dir);

#endif // DATASET_H