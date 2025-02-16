# CIFAR-10 CNN на чистом C
Ниже приведён пример максимально подробного файла документации (README.md) для проекта. В нём описывается общая идея, структура проекта, инструкция по сборке и запуску, а также подробное математическое описание работы каждого слоя нейросети.

# CIFAR CNN на чистом C

Этот проект демонстрирует реализацию простой сверточной нейронной сети (CNN) на языке C для обучения на наборе данных CIFAR (или dummy-данных). Реализация включает:
- Сверточный слой с валидной свёрткой,
- Активацию ReLU,
- Max pooling,
- Полносвязный (fully-connected, FC) слой,
- Функцию softmax с кросс-энтропийной потерей.

Реализация содержит прямой (forward) и обратный (backpropagation) проходы, а также обновление параметров с использованием стохастического градиентного спуска (SGD).

---

## Содержание

- [Обзор проекта](#обзор-проекта)
- [Структура проекта](#структура-проекта)
- [Сборка и запуск](#сборка-и-запуск)
- [Архитектура модели](#архитектура-модели)
- [Математическое описание слоёв](#математическое-описание-слоёв)
  - [Сверточный слой](#сверточный-слой)
  - [ReLU](#relu)
  - [Max Pooling](#max-pooling)
  - [Полносвязный слой](#полносвязный-слой)
  - [Softmax и кросс-энтропийная потеря](#softmax-и-кросс-энтропийная-потеря)
- [Тренировка сети](#тренировка-сети)
- [Планы и улучшения](#планы-и-улучшения)

---

## Обзор проекта

Проект реализует базовую сверточную нейронную сеть, способную обрабатывать цветные изображения размером 32×32 (CIFAR) и классифицировать их по 10 классам. Основная цель – показать принцип работы сверточных нейросетей и алгоритмов обратного распространения ошибки (backpropagation) на языке C без использования специализированных библиотек.

## Датасет CIFAR-10

Набор данных CIFAR-10 состоит из 60000 цветных изображений размером 32x32 пикселя, разделенных на 10 классов по 6000 изображений в каждом. Датасет включает 50000 тренировочных и 10000 тестовых изображений.

Данные разделены на пять тренировочных пакетов и один тестовый, каждый по 10000 изображений. Тестовый пакет содержит ровно по 1000 случайно выбранных изображений из каждого класса. Тренировочные пакеты содержат оставшиеся изображения в случайном порядке, при этом некоторые пакеты могут содержать больше изображений одного класса, чем другого. В сумме тренировочные пакеты содержат ровно по 5000 изображений каждого класса.

Классы в датасете полностью взаимоисключающие. Например, нет пересечения между классами "automobile" (автомобиль) и "truck" (грузовик). Класс "automobile" включает седаны, внедорожники и подобные транспортные средства. Класс "truck" включает только большие грузовики. Ни один из этих классов не включает пикапы.

Вот классы датасета и 10 случайных изображений для каждого класса:

![CIFAR-10 классы и примеры](images/dataset.png)

---

## Структура проекта
```
cifar_cnn/
├── include/
│   ├── activations.h      # Заголовочный файл для функций активации (ReLU)
│   ├── conv_layer.h       # Заголовочный файл для сверточного слоя
│   ├── dataset.h          # Заголовочный файл для работы с данными (dummy-данные)
│   ├── fc_layer.h         # Заголовочный файл для полносвязного слоя
│   ├── maxpool_layer.h    # Заголовочный файл для max pooling слоя
│   ├── softmax.h          # Заголовочный файл для softmax и функции потерь
│   └── utils.h            # Вспомогательные функции (например, SGD)
├── src/
│   ├── activations.c      # Реализация ReLU
│   ├── conv_layer.c       # Реализация сверточного слоя
│   ├── dataset.c          # Генерация dummy-данных
│   ├── fc_layer.c         # Реализация полносвязного слоя
│   ├── main.c             # Главная программа, объединяющая все модули
│   ├── maxpool_layer.c    # Реализация max pooling
│   ├── softmax.c          # Реализация softmax и функции потерь
│   └── utils.c            # Реализация вспомогательных функций
└── Makefile               # Скрипт сборки проекта
```
Каждый модуль отвечает за свою функциональность, что облегчает сопровождение и расширение проекта.

---

## Сборка и запуск

### Требования

- Компилятор `gcc` (или другой C-компилятор)
- Make

### Сборка

В корневой директории выполните команду:

```bash
make
```
Это скомпилирует все исходные файлы и создаст исполняемый файл в каталоге bin (например, bin/cifar_cnn).

Запуск

Запустите полученный исполняемый файл:
```
./bin/cifar_cnn
```

При запуске будет выполнено обучение сети на 100 сгенерированных (dummy) изображениях в течение 5 эпох. Выводится значение функции потерь и точность по эпохам.

## Архитектура модели

Модель состоит из следующих слоёв:
	1.	Сверточный слой (ConvLayer):
На вход подается изображение размером 32×32×3. Сверточный слой с ядром 3×3 и 16 фильтрами генерирует карту признаков размером 30×30×16.
	2.	ReLU:
Функция активации, применяемая к выходу сверточного слоя.
	3.	Max Pooling (MaxPoolLayer):
Сжимает карту признаков, используя окно 2×2, что приводит к выходу размером 15×15×16.
	4.	Полносвязный слой (FCLayer):
Преобразует вектор признаков (после flatten, размер 3600) в выходной вектор с 10 элементами (по числу классов).
	5.	Softmax и кросс-энтропийная потеря:
Преобразует выход полносвязного слоя в вероятностное распределение и вычисляет функцию потерь.

## Математическое описание слоёв

### Сверточный слой

Прямой проход:
Пусть $( x )$ – входной тензор размером $((C_{in}, H_{in}, W_{in}))$, $( w )$ – ядра свёртки размером $((C_{out}, C_{in}, K, K))$, а $( b )$ – смещения (biases) для каждого фильтра. Выходной тензор $( y )$ имеет размеры:
- $( H_{out} = H_{in} - K + 1 )$
- $( W_{out} = W_{in} - K + 1 )$

Формула свёртки для выходного элемента:

$$ y_{oc}(i, j) = b_{oc} + \sum_{ic=0}^{C_{in}-1} \sum_{m=0}^{K-1} \sum_{n=0}^{K-1} x_{ic}(i+m, j+n) \cdot w_{oc, ic, m, n} $$

где:

- $( oc )$ – индекс выходного канала,
- $( ic )$ – индекс входного канала,
- $( i, j )$ – координаты на выходной карте признаков.

Обратное распространение:
Градиенты вычисляются по весам, смещениям и входным данным с использованием цепного правила. Например, градиент по весу $( w_{oc,ic,m,n} )$ определяется как:
$$ \frac{\partial L}{\partial w_{oc, ic, m, n}} = \sum_{i,j} x_{ic}(i+m, j+n) \cdot \frac{\partial L}{\partial y_{oc}(i, j)} $$

Аналогичным образом вычисляются градиенты по входу и смещениям.

### ReLU

Функция активации ReLU применяется поэлементно:
$$ f(x) = \max(0, x) $$
Её производная:
    $$ f'(x) = \begin{cases} 1, & \text{если } x > 0 \ 0, & \text{иначе} \end{cases} $$
Это означает, что при обратном проходе градиент передаётся только через те нейроны, где $( x > 0 )$.

### Max Pooling

Max pooling применяется для уменьшения размерности карты признаков. Для каждого канала и для каждого окна размером $( p \times p )$ (в нашем случае $(2 \times 2)$ ) вычисляется:
$$ y_{c}(i, j) = \max { x_{c}(p, q) \mid p \in [i \cdot p, i \cdot p + p - 1],; q \in [j \cdot p, j \cdot p + p - 1] } $$   
При обратном распространении градиент передаётся только тому элементу, который дал максимум в каждом окне.  

При обратном распространении градиент передаётся только тому элементу, который дал максимум в каждом окне.

### Полносвязный слой

Полносвязный слой принимает на вход вектор $( x )$ (flattened из предыдущего слоя) и вычисляет:
$$ y = W x + b $$   
где:

- $( W )$ – матрица весов размером $((N_{\text{out}}, N_{\text{in}}))$
- $( b )$ – вектор смещений размером $( N_{\text{out}} )$.

Каждый элемент выходного вектора:
$$ y_i = b_i + \sum_{j=1}^{N_{\text{in}}} W_{ij} x_j $$

Обратное распространение вычисляет градиенты по весам и смещениям:
$$ \frac{\partial L}{\partial W_{ij}} = \frac{\partial L}{\partial y_i} \cdot x_j,\quad \   frac{\partial L}{\partial b_i} = \frac{\partial L}{\partial y_i} $$

### Softmax и кросс-энтропийная потеря

Softmax:
Для вектора входов $( z )$ softmax вычисляется по формуле:
$$ \sigma(z)i = \frac{e^{z_i}}{\sum{j=1}^{N} e^{z_j}} $$
где $( N )$ – число классов.

Кросс-энтропийная потеря:
При наличии истинной метки $( t )$ функция потерь вычисляется как:
$$ L = -\ln \big(\sigma(z)t\big) $$
Обратное распространение для softmax с кросс-энтропией дает градиент:
$$ \frac{\partial L}{\partial z_i} = \sigma(z)i - \mathbb{1}{{i = t}} $$
где $( \mathbb{1}{{i = t}} )$ – индикатор истинной метки (равен 1, если $( i = t )$ , иначе 0).

## Тренировка сети

Процесс обучения происходит по следующему алгоритму:
1. Прямой проход:
- Применяется сверточный слой, затем ReLU, затем max pooling.
- Выход преобразуется в вектор и подается на полносвязный слой.
- Softmax преобразует выход в вероятностное распределение, после чего вычисляется кросс-энтропийная потеря.
2.	Обратный проход (Backpropagation):
- Вычисляются градиенты от softmax к полносвязному слою.
- Градиенты передаются через обратное распространение по полносвязному, pooling, ReLU и сверточному слоям.
3.	Обновление параметров:
Параметры обновляются с использованием стохастического градиентного спуска (SGD):
$$ \theta \leftarrow \theta - \eta \cdot \frac{\partial L}{\partial \theta} $$
где $( \theta )$ – параметр модели (вес или смещение), а $( \eta )$ – скорость обучения.

## Планы и улучшения
- Загрузка реальных данных CIFAR:
В текущей реализации используются dummy-данные. Можно реализовать загрузку реального набора данных.
- Батчевое обучение:
Вместо обновления параметров после каждого примера можно использовать батчи для более стабильного обучения.
- Регуляризация и оптимизаторы:
Добавление методов регуляризации (например, dropout) и более сложных оптимизаторов (Adam, RMSprop).
- Оптимизация по скорости:
Реализация оптимизированных версий операций (например, с использованием SIMD или GPU).

# Заключение

Этот проект демонстрирует основные принципы работы сверточных нейросетей и реализации алгоритма обратного распространения ошибки на чистом C. Он служит хорошей отправной точкой для изучения и дальнейшего расширения функциональности моделей глубокого обучения.
