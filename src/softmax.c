#include "softmax.h"
#include <math.h>
#include <stdio.h>

float softmax(float *input, int length, int label, float *probabilities) {
    float max_val = input[0];
    for (int i = 1; i < length; i++) {
        if (input[i] > max_val)
            max_val = input[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < length; i++) {
        probabilities[i] = exp(input[i] - max_val);
        sum += probabilities[i];
    }
    for (int i = 0; i < length; i++) {
        probabilities[i] /= sum;
    }
    float loss = -log(probabilities[label] + 1e-7);
    return loss;
}

void softmax_backward(float *probabilities, int label, int length, float *dout) {
    for (int i = 0; i < length; i++) {
        dout[i] = probabilities[i];
    }
    dout[label] -= 1.0f;
}