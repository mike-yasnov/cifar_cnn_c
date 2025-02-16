#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

void relu_forward(float *input, float *output, int size);
void relu_backward(float *input, float *dout, float *dinput, int size);

#endif 