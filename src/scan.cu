#include "scan.cuh"

#define checkCudaError(o, l) _checkCudaError(o, l, __func__)\

int THREADS_PER_BLOCK = 512;
int ELEMENTS_PER_BLOCK = THREADS_PER_BLOCK * 2;

int nextPowerOfTwo(int x) {
	int power = 1;
	while (power < x) {
		power *= 2;
	}
	return power;
}

void scan(int *output, int *input, int length) {

    if (length > ELEMENTS_PER_BLOCK) {
		scanLargeDeviceArray(output, input, length);
	}
	else {
		scanSmallDeviceArray(output, input, length);
	}
}

void scanLargeDeviceArray(int *d_out, int *d_in, int length) {
	int remainder = length % (ELEMENTS_PER_BLOCK);
	if (remainder == 0) {
		scanLargeEvenDeviceArray(d_out, d_in, length);
	}
	else {
		// perform a large scan on a compatible multiple of elements
		int lengthMultiple = length - remainder;
        scanLargeEvenDeviceArray(d_out, d_in, lengthMultiple);
		// scan the remaining elements and add the (inclusive) last element of the large scan to this
		int *startOfOutputArray = &(d_out[lengthMultiple]);
		scanSmallDeviceArray(startOfOutputArray, &(d_in[lengthMultiple]), remainder);

        add<<<1, remainder>>>(startOfOutputArray, remainder, &(d_in[lengthMultiple - 1]), &(d_out[lengthMultiple - 1]));
	}
}

void scanSmallDeviceArray(int *d_out, int *d_in, int length) {
	int powerOfTwo = nextPowerOfTwo(length);
	prescan_arbitrary <<<1, (length + 1) / 2, 2 * powerOfTwo * sizeof(int) >>>(d_out, d_in, length, powerOfTwo);
}

void scanLargeEvenDeviceArray(int *d_out, int *d_in, int length) {
	const int blocks = length / ELEMENTS_PER_BLOCK;
	const int sharedMemArraySize = ELEMENTS_PER_BLOCK * sizeof(int);

	int *d_sums, *d_incr;
	cudaMalloc((void **)&d_sums, blocks * sizeof(int));
	cudaMalloc((void **)&d_incr, blocks * sizeof(int));


    prescan_large<<<blocks, THREADS_PER_BLOCK, 2 * sharedMemArraySize>>>(d_out, d_in, ELEMENTS_PER_BLOCK, d_sums);


    const int sumsArrThreadsNeeded = (blocks + 1) / 2;

    if (sumsArrThreadsNeeded > THREADS_PER_BLOCK) {
        scanLargeDeviceArray(d_incr, d_sums, blocks);
    }
    else {
        scanSmallDeviceArray(d_incr, d_sums, blocks);
    }

    add<<<blocks, ELEMENTS_PER_BLOCK>>>(d_out, ELEMENTS_PER_BLOCK, d_incr);

	cudaFree(d_sums);
	cudaFree(d_incr);
}