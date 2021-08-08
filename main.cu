#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <algorithm>
#include <unistd.h>

#define max(a, b) (a > b ? a : b)
#define min(a, b) (a < b ? a : b)

// Max threadsize is 1024 32*32
typedef unsigned char ubyte;

void printWorld(ubyte *world, uint size);
void zeroWorld(ubyte *world, uint size);
void copy(ubyte *pattern, int patternsize, ubyte *world, uint size);
int coords(int x, int y, int size);

__global__ void gameoflifeturn(ubyte *inworld, ubyte *bufferworld, short size)
{

    // We need to find the x,y of the cell we are looking at
    // Because this is a 1d array we have to do some maths.
    uint x = threadIdx.x + (blockDim.x * blockIdx.x);
    uint y = threadIdx.y + (blockDim.y * blockIdx.y);
    //printf("(%d,%d) (%d %d) (%d %d) (%d, %d) (%d, %d)\n", x, y, threadIdx.x, threadIdx.y, blockDim.x, blockDim.y, blockIdx.x, blockIdx.y, gridDim.x, gridDim.y);
    uint xLeft = ((x + size) - 1) % size;
    uint xRight = (x + 1) % size;
    uint yUp = (y + 1) % size;
    uint yDown = (y + size - 1) % size;
    uint yAbs = size * y;
    uint yUpAbs = yUp * size;
    uint yDownAbs = yDown * size;

    uint Abs = x + yAbs;
    uint aliveCells = inworld[xLeft + yUpAbs] + inworld[x + yUpAbs] + inworld[xRight + yUpAbs] + inworld[xLeft + yAbs] + inworld[xRight + yAbs] + inworld[xLeft + yDownAbs] + inworld[x + yDownAbs] + inworld[xRight + yDownAbs];

    //Any live cell with two or three live neighbours survives.
    //Any dead cell with three live neighbours becomes a live cell.
    //All other live cells die in the next generation. Similarly, all other dead cells stay dead.
    bufferworld[Abs] = aliveCells == 3 || (aliveCells == 2 && inworld[Abs]) ? 1 : 0;
}

int main()
{
    // To keep the math easy the size of the world must be a square of a square, i.e. X^2^2
    // This is so we can easily divide up the world into square blocks for processing
    // To make it even easier size should be a poer of 2, i.e. 2^X
    uint size = 256;
    int turns = 100000;

    uint ncells = size * size;

    // With the max number of threads being 1024
    // The number of threads here will describe the number of blocks
    uint threadsCount = min(ncells, 1024);
    uint threadDimSize = sqrt(threadsCount);

    // Threads create a block of sqrt(threadCount)^2
    dim3 threadsPerBlock(threadDimSize, threadDimSize);

    // Now we need to find the number of blocks this is the size/ThreadDimSize
    uint blockDimSize = size / threadDimSize;
    dim3 numBlocks(blockDimSize, blockDimSize);

    // Lets make sure our math is correct
    // The number of cells is a multiple of threadcount
    assert(ncells % threadsCount == 0);
    // the number of blocks * num of threads = size
    assert(blockDimSize * threadDimSize == size);

    printf("Size %d, ncells %d, Threads: %d, ThreadDimSize %d, BlockDimSize %d\n", size, ncells, threadsCount, threadDimSize, blockDimSize);

    // We make a 1d array of bytes, ehere each byte is a cell, to describe the world
    ubyte *world;
    int worldSize = sizeof(ubyte) * ncells;
    world = (ubyte *)malloc(worldSize);

    // We setup the world by first zeroing it out, then copying a pattern (this is the glider)
    zeroWorld(world, size);
    ubyte pattern[5][5] = {
        {0, 0, 0, 1, 0},
        {0, 1, 0, 1, 0},
        {0, 0, 1, 1, 0},
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0},
    };
    copy((ubyte *)pattern, 5, world, size);
    // printWorld(world, size);

    // Set up the Device Memory by create the world and a buffer
    // Then by Mallocing on the device, then copying the world over to the device
    ubyte *inworld, *bufferworld;
    cudaMalloc((void **)&inworld, worldSize);
    cudaMalloc((void **)&bufferworld, worldSize);
    cudaMemcpy(inworld, world, worldSize, cudaMemcpyHostToDevice);

    // Time some stuff
    struct timeval t0, t1;
    gettimeofday(&t0, NULL);

    // Run the world
    int turn;
    for (turn = 0; turn < turns; turn++)
    {
        gameoflifeturn<<<numBlocks, threadsPerBlock>>>(inworld, bufferworld, size);
        std::swap(inworld, bufferworld);
    }

    // Finish timing
    gettimeofday(&t1, NULL);

    // Copy the value of the world back to host memory
    cudaMemcpy(world, inworld, worldSize, cudaMemcpyDeviceToHost);

    // system("clear");
    // printWorld(world, size);

    // How many seconds it took to execute
    float seconds = t1.tv_sec - t0.tv_sec + 1E-6 * (t1.tv_usec - t0.tv_usec);
    // How many total calculations
    float MMcellCalculations = (1.0 * turns * ncells) / 1000000;
    // Millions of Calculations per second
    float MMcellsCalculatedperSecond = MMcellCalculations / seconds;

    printf("Did %f million cells per second in %f\n", MMcellsCalculatedperSecond, seconds);

    // Free all the Device and host memory
    cudaFree(inworld);
    cudaFree(bufferworld);
    free(world);

    return 0;
}

void copy(ubyte *pattern, int patternsize, ubyte *world, uint size)
{

    ubyte x, y;
    for (y = 0; y < patternsize; y++)
    {
        for (x = 0; x < patternsize; x++)
        {
            world[x + (size * y)] = pattern[x + (y * patternsize)];
        }
    }
}

void zeroWorld(ubyte *world, uint size)
{
    int x, y;

    for (y = 0; y < size; ++y)
    {
        for (x = 0; x < size; ++x)
        {
            world[x + (y * size)] = 0;
        }
    }
}

void printWorld(ubyte *world, uint size)
{
    int x, y;

    printf("   ------   \n");
    for (y = 0; y < size; y++)
    {
        for (x = 0; x < size; x++)
        {
            printf("%d", world[x + (y * size)]);
        }
        printf("\n");
    }
    printf("   ------   \n\n");
}
