#ifndef CONV_LAYER_H
#define CONV_LAYER_H

typedef struct {
    int in_channels;
    int out_channels;
    int kernel_size;
    int in_height;
    int in_width;
    int out_height;
    int out_width;
    float *weights;    
    float *biases;    
    float *input;      
    float *output;     
    float *dweights;   
    float *dbiases;    
} ConvLayer;

void conv_forward(ConvLayer *layer, float *input);
void conv_backward(ConvLayer *layer, float *dout, float *dinput);

#endif 