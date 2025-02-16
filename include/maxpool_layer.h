#ifndef MAXPOOL_LAYER_H
#define MAXPOOL_LAYER_H

typedef struct {
    int channels;
    int in_height;
    int in_width;
    int pool_size;
    int out_height;
    int out_width;
    float *input;     
    float *output;    
    int *max_index;   
} MaxPoolLayer;

void maxpool_forward(const MaxPoolLayer *pool, const float *input);
void maxpool_backward(MaxPoolLayer *pool, float *dout, float *dinput);

#endif 