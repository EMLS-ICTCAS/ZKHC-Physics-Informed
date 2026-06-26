# Description: Makefile for the project

# Compiler and flags
CXX := g++
CC := gcc
CXXFLAGS := -Wall -Wextra -Wpedantic -Wshadow -Wformat=2 -Wcast-align -Wconversion -Wsign-conversion -Wnull-dereference -g3 -Ofast -std=c++11
CFLAGS := -Wall -Wextra -pedantic -Ofast -std=c11

# Libraries and paths
HDF5_DIR := /usr/include/hdf5/serial/
HDF5_LIB_DIR := /usr/lib/x86_64-linux-gnu/hdf5/serial
EIGEN_DIR := /usr/include/eigen3
LIBS := -lm -L$(HDF5_LIB_DIR) -lhdf5_cpp -lhdf5
INCLUDES := -I$(EIGEN_DIR) -I$(HDF5_DIR)

# Folders
SRC := src
INCLUDE := include
BIN := bin

# Executable name
TARGET := main

# Sources and objects
SOURCES := $(wildcard $(SRC)/*.c) $(wildcard $(SRC)/*.cpp)
OBJECTS := $(patsubst $(SRC)/%.cpp, $(BIN)/%.o, $(filter %.cpp, $(SOURCES))) $(patsubst $(SRC)/%.c, $(BIN)/%.o, $(filter %.c, $(SOURCES))) 

# Headers
HEADERS := $(wildcard $(INCLUDE)/*.h) $(wildcard $(INCLUDE)/*.hpp)

.PHONY: all clean

all: $(BIN)/$(TARGET)

$(BIN)/$(TARGET): $(OBJECTS)
	@$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

$(BIN)/%.o: $(SRC)/%.cpp $(HEADERS)
	@$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BIN)/%.o: $(SRC)/%.c $(HEADERS)
	@$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

clean:
	@rm -f $(BIN)/*.o $(BIN)/$(TARGET)
