CC = gcc
CFLAGS = -Wall -O2
INCLUDES = -Iinclude
SRCDIR = src
OBJDIR = obj
BINDIR = bin

COMMON_SRCS = $(SRCDIR)/activations.c $(SRCDIR)/conv_layer.c $(SRCDIR)/dataset.c \
              $(SRCDIR)/fc_layer.c $(SRCDIR)/maxpool_layer.c $(SRCDIR)/softmax.c $(SRCDIR)/utils.c

TRAIN_SRCS = $(SRCDIR)/train.c
PREDICT_SRCS = $(SRCDIR)/predict.c

COMMON_OBJS = $(patsubst $(SRCDIR)/%.c,$(OBJDIR)/%.o,$(COMMON_SRCS))
TRAIN_OBJS = $(patsubst $(SRCDIR)/%.c,$(OBJDIR)/%.o,$(TRAIN_SRCS))
PREDICT_OBJS = $(patsubst $(SRCDIR)/%.c,$(OBJDIR)/%.o,$(PREDICT_SRCS))

all: $(BINDIR)/train $(BINDIR)/predict

$(BINDIR)/train: $(COMMON_OBJS) $(TRAIN_OBJS)
	mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $^

$(BINDIR)/predict: $(COMMON_OBJS) $(PREDICT_OBJS)
	mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $^

$(OBJDIR)/%.o: $(SRCDIR)/%.c
	mkdir -p $(OBJDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -rf $(OBJDIR) $(BINDIR)