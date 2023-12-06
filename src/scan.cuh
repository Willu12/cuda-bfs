#pragma once
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kernels.cuh"

void blockscan(int *output, int *input, int length);
void scan(int *output, int *input, int length);

void scanLargeDeviceArray(int *output, int *input, int length);
void scanSmallDeviceArray(int *d_out, int *d_in, int length);
void scanLargeEvenDeviceArray(int *output, int *input, int length);