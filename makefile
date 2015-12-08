CC = gcc
FLAGS = -fpermissive
LIBFLAGS = -lm -ldl -lGL -lGLU -lglut
OUTPUT = drd.o
SOURCE = drd.c

compile: $(SOURCE) clean
	@$(CC) $(SOURCE) $(FLAGS) -o $(OUTPUT) $(LIBFLAGS)

clean:
	@rm -f $(OUTPUT)

run: $(OUTPUT)
	@./$(OUTPUT)