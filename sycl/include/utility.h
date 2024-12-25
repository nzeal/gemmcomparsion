#ifndef UTILITY_H
#define UTILITY_H

#include <cstdlib>
#include <iostream>
#include <algorithm>

template<typename T>
inline void allocateMemory(T** A, T** B, T** C, size_t size) {
    *A = new T[size];
    *B = new T[size];
    *C = new T[size];
    if (*A == nullptr || *B == nullptr || *C == nullptr) {
        std::cerr << "Memory allocation failed" << std::endl;
        exit(1);
    }
}

template<typename T>
inline void freeMemory(T* A, T* B, T* C) {
    delete[] A;
    delete[] B;
    delete[] C;
}

template <typename T>
void printArray(const T* array, size_t size) {
    for (size_t i = 0; i < std::min(size, static_cast<size_t>(5)); ++i) {
        std::cout << array[i] << " ";
    }
    std::cout << std::endl;
}

#endif // UTILITY_H
