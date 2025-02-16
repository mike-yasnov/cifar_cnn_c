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

void update_parameters(float *params, float *grads, int size, float learning_rate) {
    for (int i = 0; i < size; i++) {
        params[i] -= learning_rate * grads[i];
    }
}

int save_weights(const char *filename, float *weights, int size) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        printf("Ошибка: Не удалось открыть файл %s для записи\n", filename);
        return -1;
    }

    // Сначала записываем размер массива
    if (fwrite(&size, sizeof(int), 1, fp) != 1) {
        printf("Ошибка: Не удалось записать размер весов\n");
        fclose(fp);
        return -1;
    }

    // Затем записываем сами веса
    if (fwrite(weights, sizeof(float), size, fp) != size) {
        printf("Ошибка: Не удалось записать веса\n");
        fclose(fp);
        return -1;
    }

    fclose(fp);
    printf("Веса успешно сохранены в файл %s\n", filename);
    return 0;
}

int load_weights(const char *filename, float *weights, int size) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        printf("Ошибка: Не удалось открыть файл %s для чтения\n", filename);
        return -1;
    }

    // Читаем размер весов из файла
    int saved_size;
    if (fread(&saved_size, sizeof(int), 1, fp) != 1) {
        printf("Ошибка: Не удалось прочитать размер весов\n");
        fclose(fp);
        return -1;
    }

    // Проверяем соответствие размеров
    if (saved_size != size) {
        printf("Ошибка: Размер сохраненных весов (%d) не соответствует ожидаемому размеру (%d)\n", 
               saved_size, size);
        fclose(fp);
        return -1;
    }

    // Читаем веса
    if (fread(weights, sizeof(float), size, fp) != size) {
        printf("Ошибка: Не удалось прочитать веса\n");
        fclose(fp);
        return -1;
    }

    fclose(fp);
    printf("Веса успешно загружены из файла %s\n", filename);
    return 0;
}

const char* get_class_name(int class_id) {
    static const char *class_names[] = {
        "самолет", "автомобиль", "птица", "кошка", "олень",
        "собака", "лягушка", "лошадь", "корабль", "грузовик"
    };
    
    if (class_id >= 0 && class_id < 10) {
        return class_names[class_id];
    }
    return "неизвестный класс";
}

void save_prediction_result(const char *filename, float *image, int width, int height, int channels, 
                          const char *predicted_class) {
    // Создаем имя файла для изображения, заменяя расширение .txt на .png
    char image_filename[256];
    strncpy(image_filename, filename, sizeof(image_filename) - 1);
    char *dot = strrchr(image_filename, '.');
    if (dot) {
        strcpy(dot, ".png");
    }

    // Создаем буфер для RGB изображения
    unsigned char *rgb_image = (unsigned char*)malloc(width * height * 3);
    
    // Преобразуем float значения [0,1] в unsigned char [0,255]
    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            for (int c = 0; c < 3; c++) {
                float pixel = image[c * height * width + h * width + w];
                // Ограничиваем значения от 0 до 1
                pixel = pixel < 0 ? 0 : (pixel > 1 ? 1 : pixel);
                // Преобразуем в диапазон 0-255
                rgb_image[(h * width + w) * 3 + c] = (unsigned char)(pixel * 255);
            }
        }
    }

    // Сохраняем изображение
    if (!stbi_write_png(image_filename, width, height, 3, rgb_image, width * 3)) {
        printf("Ошибка: Не удалось сохранить изображение %s\n", image_filename);
    } else {
        printf("Изображение сохранено в файл: %s\n", image_filename);
    }

    // Сохраняем информацию о предсказании в текстовый файл
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        printf("Ошибка: Не удалось создать файл результата %s\n", filename);
        free(rgb_image);
        return;
    }

    fprintf(fp, "Предсказанный класс: %s\n", predicted_class);
    fprintf(fp, "Изображение сохранено в файл: %s\n", image_filename);

    fclose(fp);
    free(rgb_image);
    printf("Информация о предсказании сохранена в файл: %s\n", filename);
}

int model_inference(float *image, ConvLayer *conv, MaxPoolLayer *pool, FCLayer *fc,
                   float *relu_out, float *softmax_probs) {
    int conv_output_size = conv->out_channels * conv->out_height * conv->out_width;

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
           get_class_name(predicted_class), max_prob * 100);

    return predicted_class;
}