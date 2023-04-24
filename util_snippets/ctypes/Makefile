CC=gcc
CFLAGS=-Wall -Werror -fPIC
LDFLAGS=-shared

SRCS=example.c
OBJS=$(SRCS:.c=.o)
TARGET=example.so

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)