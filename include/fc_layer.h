#ifndef FC_LAYER_H
#define FC_LAYER_H

typedef struct {
    int input_size;
    int output_size;
    float *weights;   
    float *biases;    
    float *input;     
    float *output;    
    float *dweights;  
    float *dbiases;   
} FCLayer;

void fc_forward(FCLayer *fc, float *input);
void fc_backward(FCLayer *fc, float *dout, float *dinput);

#endif 