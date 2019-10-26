CC=g++
CFLAGS=-g -Wall -std=c++14
EIGEN_PATH=./submodule/eigen
SRC= ./src
LIB=./lib
build:
	$(CC) -o o.out $(SRC)/main.cc $(lib)/solver.cc -I $(EIGEN_PATH)