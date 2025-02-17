#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "utils.h"
#include "conv_layer.h"
#include "fc_layer.h"
#include "maxpool_layer.h"
#include "activations.h"
#include "softmax.h"

#define NUM_CLASSES 10
#define MAX_FILENAME_LENGTH 256
#define RGB_CHANNELS 3
#define MAX_IMAGE_SIZE 1048576  // 1024*1024 для безопасности

void model_update_parameters(float *params, float *grads, int size, float learning_rate) {
    for (int i = 0; i < size; i++) {
        params[i] -= learning_rate * grads[i];
    }
}

int model_save_weights(const char *filename, const float *weights, int size) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Ошибка: Не удалось открыть файл %s для записи\n", filename);
        return -1;
    }

    // Сначала записываем размер массива
    if (fwrite(&size, sizeof(int), 1, fp) != 1) {
        fprintf(stderr, "Ошибка: Не удалось записать размер весов\n");
        fclose(fp);
        return -1;
    }

    // Затем записываем сами веса
    if (fwrite(weights, sizeof(float), size, fp) != size) {
        fprintf(stderr, "Ошибка: Не удалось записать веса\n");
        fclose(fp);
        return -1;
    }

    fclose(fp);
    printf("Веса успешно сохранены в файл %s\n", filename);
    return 0;
}

int model_load_weights(const char *filename, float *weights, int size) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Ошибка: Не удалось открыть файл %s для чтения\n", filename);
        return -1;
    }

    // Читаем размер весов из файла
    int saved_size;
    if (fread(&saved_size, sizeof(int), 1, fp) != 1) {
        fprintf(stderr, "Ошибка: Не удалось прочитать размер весов\n");
        fclose(fp);
        return -1;
    }

    // Проверяем соответствие размеров
    if (saved_size != size) {
        fprintf(stderr, "Ошибка: Размер сохраненных весов (%d) не соответствует ожидаемому размеру (%d)\n", 
               saved_size, size);
        fclose(fp);
        return -1;
    }

    // Читаем веса
    if (fread(weights, sizeof(float), size, fp) != size) {
        fprintf(stderr, "Ошибка: Не удалось прочитать веса\n");
        fclose(fp);
        return -1;
    }

    fclose(fp);
    printf("Веса успешно загружены из файла %s\n", filename);
    return 0;
}

const char* model_get_class_name(int class_id) {
    static const char *class_names[NUM_CLASSES] = {
        "самолет", "автомобиль", "птица", "кошка", "олень",
        "собака", "лягушка", "лошадь", "корабль", "грузовик"
    };
    
    if (class_id >= 0 && class_id < NUM_CLASSES) {
        return class_names[class_id];
    }
    return "неизвестный класс";
}

void model_save_prediction(const char *filename, const float *image, 
                         int width, int height, int channels, 
                         const char *predicted_class) {
    // Проверка входных параметров
    if (!filename || !image || !predicted_class) {
        fprintf(stderr, "Ошибка: Некорректные входные параметры\n");
        return;
    }

    if (width <= 0 || height <= 0 || channels <= 0 || 
        channels != RGB_CHANNELS || 
        (size_t)width * height > MAX_IMAGE_SIZE) {
        fprintf(stderr, "Ошибка: Некорректные размеры изображения\n");
        return;
    }

    char image_filename[MAX_FILENAME_LENGTH] = {0};
    strncpy(image_filename, filename, MAX_FILENAME_LENGTH - 1);
    char *dot = strrchr(image_filename, '.');
    if (dot) {
        strcpy(dot, ".png");
    }

    // Проверяем возможное переполнение при выделении памяти
    size_t buffer_size = (size_t)width * height * RGB_CHANNELS;
    if (buffer_size > MAX_IMAGE_SIZE) {
        fprintf(stderr, "Ошибка: Слишком большой размер изображения\n");
        return;
    }

    // Создаем буфер для RGB изображения
    unsigned char *rgb_image = malloc(buffer_size);
    if (!rgb_image) {
        fprintf(stderr, "Ошибка: Не удалось выделить память для изображения\n");
        return;
    }
    
    // Преобразуем float значения [0,1] в unsigned char [0,255]
    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            for (int c = 0; c < RGB_CHANNELS; c++) {
                const float pixel = image[c * height * width + h * width + w];
                // Ограничиваем значения от 0 до 1
                const float clamped_pixel = pixel < 0 ? 0 : (pixel > 1 ? 1 : pixel);
                // Преобразуем в диапазон 0-255
                rgb_image[(h * width + w) * RGB_CHANNELS + c] = (unsigned char)(clamped_pixel * 255);
            }
        }
    }

    // Сохраняем изображение
    if (!stbi_write_png(image_filename, width, height, RGB_CHANNELS, rgb_image, width * RGB_CHANNELS)) {
        fprintf(stderr, "Ошибка: Не удалось сохранить изображение %s\n", image_filename);
        free(rgb_image);
        return;
    }
    printf("Изображение сохранено в файл: %s\n", image_filename);

    // Сохраняем информацию о предсказании в текстовый файл
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Ошибка: Не удалось создать файл результата %s\n", filename);
        free(rgb_image);
        return;
    }

    if (fprintf(fp, "Предсказанный класс: %s\n", predicted_class) < 0 ||
        fprintf(fp, "Изображение сохранено в файл: %s\n", image_filename) < 0) {
        fprintf(stderr, "Ошибка: Не удалось записать информацию в файл %s\n", filename);
        fclose(fp);
        free(rgb_image);
        return;
    }

    fclose(fp);
    free(rgb_image);
    printf("Информация о предсказании сохранена в файл: %s\n", filename);
}

int model_inference(const float *image, const ConvLayer *conv, const MaxPoolLayer *pool, 
                   const FCLayer *fc, float *relu_out, float *softmax_probs) {
    // Проверка входных параметров
    if (!image || !conv || !pool || !fc || !relu_out || !softmax_probs) {
        fprintf(stderr, "Ошибка: Некорректные входные параметры\n");
        return -1;
    }

    // Проверка согласованности размеров
    if (conv->out_channels != pool->channels ||
        conv->out_height != pool->in_height ||
        conv->out_width != pool->in_width) {
        fprintf(stderr, "Ошибка: Несогласованные размеры слоев\n");
        return -1;
    }

    const int conv_output_size = conv->out_channels * conv->out_height * conv->out_width;
    if (conv_output_size <= 0) {
        fprintf(stderr, "Ошибка: Некорректный размер выхода сверточного слоя\n");
        return -1;
    }

    // Прямой проход через сеть
    conv_forward(conv, image);
    relu_forward(conv->output, relu_out, conv_output_size);
    maxpool_forward(pool, relu_out);
    memcpy(fc->input, pool->output, sizeof(float) * fc->input_size);
    fc_forward(fc, fc->input);
    
    // Вычисляем softmax без метки (используем фиктивную метку 0)
    softmax(fc->output, fc->output_size, 0, softmax_probs);

    // Находим класс с максимальной вероятностью
    int predicted_class = 0;
    float max_prob = softmax_probs[0];
    for (int i = 1; i < fc->output_size; i++) {
        if (softmax_probs[i] > max_prob) {
            max_prob = softmax_probs[i];
            predicted_class = i;
        }
    }

    printf("Предсказание: %s (уверенность: %.2f%%)\n", 
           model_get_class_name(predicted_class), max_prob * 100);

    return predicted_class;
}

void model_save_prediction_result(const char *filename, const float *image, int width, int height, int channels, const char *prediction) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Ошибка: не удалось открыть файл %s для записи\n", filename);
        return;
    }

    char image_filename[MAX_FILENAME_LENGTH] = {0};
    strncpy(image_filename, filename, MAX_FILENAME_LENGTH - 1);
    char *dot = strrchr(image_filename, '.');
    if (dot) {
        strcpy(dot, ".png");
    }

    fprintf(file, "Информация о предсказании:\n");
    fprintf(file, "--------------------\n");
    fprintf(file, "Размер изображения: %dx%dx%d\n", width, height, channels);
    fprintf(file, "Предсказанный класс: %s\n", prediction);
    fprintf(file, "Изображение сохранено в: %s\n", image_filename);

    fclose(file);
}