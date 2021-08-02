#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <algorithm>
#include <unistd.h>

// Max threadsize is 1024 32*32
typedef unsigned char ubyte;

void printWorld(ubyte *board, uint size);
void zeroWorld(ubyte *board, uint size);

int coords(int x, int y, int size);

__global__ void gameoflifeturn(ubyte *inboard, ubyte *bufferboard, short size)
{

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
    uint aliveCells = inboard[xLeft + yUpAbs] + inboard[x + yUpAbs] + inboard[xRight + yUpAbs] + inboard[xLeft + yAbs] + inboard[xRight + yAbs] + inboard[xLeft + yDownAbs] + inboard[x + yDownAbs] + inboard[xRight + yDownAbs];

    //Any live cell with two or three live neighbours survives.
    //Any dead cell with three live neighbours becomes a live cell.
    //All other live cells die in the next generation. Similarly, all other dead cells stay dead.
    bufferboard[Abs] = aliveCells == 3 || (aliveCells == 2 && inboard[Abs]) ? 1 : 0;
}

int coords(int x, int y, int size)
{
    // Array to single dimension
    /*
    e.g. 3/3 should be
    x,y
    0,0 = 0
    1,0 = 1 
    2,0 = 2
    0,1 = 3
    1,1 = 4
    2,1 = 5
    0,2 = 6
    1,2 = 7
    2,2 = 8
    */
    // Wrap around
    if (x >= size)
    {
        x = x % size;
    }
    else if (x < 0)
    {
        x = ((size * 10) + x) % size;
    }

    if (y >= size)
    {
        y = y % size;
    }
    else if (y < 0)
    {
        y = ((size * 10) + y) % size;
    }

    return x + (size * y);
}

void o1(ubyte *board, uint size)
{
    const short o1[5][5] = {
        {0, 0, 0, 1, 0},
        {0, 1, 0, 1, 0},
        {0, 0, 1, 1, 0},
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0},
    };
    short x, y;
    for (y = 0; y < 5; y++)
    {
        for (x = 0; x < 5; x++)
        {
            board[coords(x, y, size)] = o1[y][x];
        }
    }
}

__global__ void simpleLifeKernel(const ubyte *inboard, ubyte *bufferboard, uint size)
{
    uint worldSize = size * size;

    for (uint cellId = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
         cellId < worldSize;
         cellId += blockDim.x * gridDim.x)
    {
        uint x = cellId % size;
        uint yAbs = cellId - x;
        uint xLeft = (x + size - 1) % size;
        uint xRight = (x + 1) % size;
        uint yAbsUp = (yAbs + worldSize - size) % worldSize;
        uint yAbsDown = (yAbs + size) % worldSize;

        uint aliveCells = inboard[xLeft + yAbsUp] + inboard[x + yAbsUp] + inboard[xRight + yAbsUp] + inboard[xLeft + yAbs] + inboard[xRight + yAbs] + inboard[xLeft + yAbsDown] + inboard[x + yAbsDown] + inboard[xRight + yAbsDown];

        bufferboard[x + yAbs] =
            aliveCells == 3 || (aliveCells == 2 && inboard[x + yAbs]) ? 1 : 0;
    }
}

int main()
{
    // To keep the math easy (for reasons we will see) the size must be a square of squares, i.e. X^2^2
    // so the value must be , so the easiest numbers to deal with are 2^n values
    uint size = 64;
    uint ncells = size * size;
    // The max number of threads is 1024, it must be a square number, e.g. X^2 and sqrt()
    // The number of threads here will describe the number of blocks
    uint threadsCount = 1024;
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

    // boardposition = width*heigth cudaboard[width*height] == position
    ubyte *board;
    int boardSize = sizeof(ubyte) * ncells;
    board = (ubyte *)malloc(boardSize);

    // Setup the board
    zeroWorld(board, size);
    o1(board, size);
    printWorld(board, size);

    // Set up the Device Memory
    ubyte *inboard, *bufferboard;
    cudaMalloc((void **)&inboard, boardSize);
    cudaMalloc((void **)&bufferboard, boardSize);
    cudaMemcpy(inboard, board, boardSize, cudaMemcpyHostToDevice);

    // Time some stuff
    struct timeval t0, t1;
    gettimeofday(&t0, NULL);
    int turn, turns = 10000;
    for (turn = 0; turn < turns; turn++)
    {
        gameoflifeturn<<<numBlocks, threadsPerBlock>>>(inboard, bufferboard, size);
        std::swap(inboard, bufferboard);
    }
    gettimeofday(&t1, NULL);

    cudaMemcpy(board, inboard, boardSize, cudaMemcpyDeviceToHost);
    system("clear");
    printWorld(board, size);

    float seconds = t1.tv_sec - t0.tv_sec + 1E-6 * (t1.tv_usec - t0.tv_usec);
    int worldsize = size * size;
    float MM = (turns * 1.0) * worldsize;
    float MMcellsCalculated = MM / 1000000000;
    float total = MMcellsCalculated / seconds;
    printf("Did %f billion cells per second in %f\n", total, seconds);

    cudaFree(inboard);
    cudaFree(bufferboard);
    free(board);

    return 0;
}

void zeroWorld(ubyte *board, uint size)
{
    int x, y;

    for (y = 0; y < size; ++y)
    {
        for (x = 0; x < size; ++x)
        {
            board[coords(x, y, size)] = 0;
        }
    }
}

void printWorld(ubyte *board, uint size)
{
    int x, y;

    printf("   ------   \n");
    for (y = 0; y < size; y++)
    {
        for (x = 0; x < size; x++)
        {
            printf("%d", board[coords(x, y, size)]);
        }
        printf("\n");
    }
    printf("   ------   \n\n");
}
