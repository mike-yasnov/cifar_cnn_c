#include "activations.h"

void relu_forward(float *input, float *output, int size) {
    for (int i = 0; i < size; i++) {
        output[i] = (input[i] > 0) ? input[i] : 0.0f;
    }
}

void relu_backward(float *input, float *dout, float *dinput, int size) {
    for (int i = 0; i < size; i++) {
        dinput[i] = (input[i] > 0) ? dout[i] : 0.0f;
    }
}