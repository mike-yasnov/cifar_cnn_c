#include "utils.h"

void update_parameters(float *params, float *grads, int size, float learning_rate) {
    for (int i = 0; i < size; i++) {
        params[i] -= learning_rate * grads[i];
    }
}