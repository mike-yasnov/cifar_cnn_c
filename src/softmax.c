#include "softmax.h"
#include <math.h>
#include <stdio.h>

#define SOFTMAX_EPS 1e-7f

float softmax(const float *input, int length, int label, float *probabilities) {
    if (!input || !probabilities || length <= 0 || label < 0 || label >= length) {
        fprintf(stderr, "Ошибка: Некорректные входные параметры в softmax\n");
        return -1.0f;
    }

    float max_val = input[0];
    for (int i = 1; i < length; i++) {
        if (input[i] > max_val)
            max_val = input[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < length; i++) {
        const float exp_val = exp(input[i] - max_val);
        if (isinf(exp_val)) {
            fprintf(stderr, "Ошибка: Переполнение в вычислении экспоненты\n");
            return -1.0f;
        }
        probabilities[i] = exp_val;
        sum += probabilities[i];
    }

    if (sum < SOFTMAX_EPS) {
        fprintf(stderr, "Ошибка: Слишком маленькая сумма в softmax\n");
        return -1.0f;
    }

    for (int i = 0; i < length; i++) {
        probabilities[i] /= sum;
    }

    float loss = -log(probabilities[label] + SOFTMAX_EPS);
    return loss;
}

void softmax_backward(const float *probabilities, int label, int length, float *dout) {
    if (!probabilities || !dout || length <= 0 || label < 0 || label >= length) {
        fprintf(stderr, "Ошибка: Некорректные входные параметры в softmax_backward\n");
        return;
    }

    for (int i = 0; i < length; i++) {
        dout[i] = probabilities[i];
    }
    dout[label] -= 1.0f;
}