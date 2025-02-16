#include "activations.h"
#include <stdio.h>

void relu_forward(const float *input, float *output, int size) {
    if (!input || !output || size <= 0) {
        fprintf(stderr, "Ошибка: Некорректные входные параметры в relu_forward\n");
        return;
    }
    for (int i = 0; i < size; i++) {
        output[i] = (input[i] > 0) ? input[i] : 0.0f;
    }
}

void relu_backward(const float *input, const float *dout, float *dinput, int size) {
    if (!input || !dout || !dinput || size <= 0) {
        fprintf(stderr, "Ошибка: Некорректные входные параметры в relu_backward\n");
        return;
    }
    for (int i = 0; i < size; i++) {
        dinput[i] = (input[i] > 0) ? dout[i] : 0.0f;
    }
}