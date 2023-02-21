#include <Windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void matrixAddition(int rows, int cols, int *A, int *B, int *C) {
  int i, j;
  for (i = 0; i < rows; i++) {
    for (j = 0; j < cols; j++) {
      C[i * cols + j] = A[i * cols + j] + B[i * cols + j];
    }
  }
}

int main() {
  const int rows = 100;
  const int cols = 100;
  
  int *A = (int*)malloc(rows * cols * sizeof(int));
  int *B = (int*)malloc(rows * cols * sizeof(int));
  int *C = (int*)malloc(rows * cols * sizeof(int));

  // Initialize A and B with values
  srand(time(NULL));
  int i, j;
  for (i = 0; i < rows; i++) {
    for (j = 0; j < cols; j++) {
      A[i * cols + j] = rand();
      B[i * cols + j] = rand();
    }
  }

  // Windows implementation of clock()
  LARGE_INTEGER frequency;
  QueryPerformanceFrequency(&frequency);

  LARGE_INTEGER start;
  QueryPerformanceCounter(&start);
  matrixAddition(rows, cols, A, B, C);
  LARGE_INTEGER end;
  QueryPerformanceCounter(&end);
  
  // for (i = 0; i < rows; i++) {
  //   for (j = 0; j < cols; j++) {
  //     printf("%d ", C[i * cols + j]);
  //   }
  //   printf("\n");
  // }

  double elapsed = (double)(end.QuadPart - start.QuadPart) / frequency.QuadPart;
  printf("Execution time: %.2f seconds\n", elapsed);
  printf("Execution time: %.2f milliseconds\n", elapsed * 1000);
  printf("Execution time: %.2f microseconds\n", elapsed * 1000000);
  
  free(A);
  free(B);
  free(C);
  
  return 0;
}