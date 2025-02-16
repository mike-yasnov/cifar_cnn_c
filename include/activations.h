#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

void relu_forward(const float *input, float *output, int size);
void relu_backward(const float *input, const float *dout, float *dinput, int size);

#endif 