#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <algorithm>

// Max threadsize is 1024 32*32
typedef unsigned char ubyte;

void printWorld(ubyte *board, uint size);
void zeroWorld(ubyte *board, uint size);

int coords(int x, int y, int size);

__global__ void playTurn(ubyte *inboard, ubyte *bufferboard, short size)
{
    short x = threadIdx.x;
    short y = threadIdx.y;

    short xLeft = ((x + size) - 1) % size;
    short xRight = (x + 1) % size;
    short yUp = (y + 1) % size;
    short yDown = (y + size - 1) % size;
    short yAbs = size * y;
    short yUpAbs = yUp * size;
    short yDownAbs = yDown * size;

    short Abs = x + yAbs;
    short aliveCells = inboard[xLeft + yUpAbs] + inboard[x + yUpAbs] + inboard[xRight + yUpAbs] + inboard[xLeft + yAbs] + inboard[xRight + yAbs] + inboard[xLeft + yDownAbs] + inboard[x + yDownAbs] + inboard[xRight + yDownAbs];

    //Any live cell with two or three live neighbours survives.
    //Any dead cell with three live neighbours becomes a live cell.
    //All other live cells die in the next generation. Similarly, all other dead cells stay dead.
    bufferboard[Abs] = aliveCells == 3 || (aliveCells == 2 && inboard[Abs]) ? 1 : 0;
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

int main()
{
    uint SIZE = 512;

    // boardposition = width*heigth cudaboard[width*height] == position
    ubyte *board;
    int boardSize = sizeof(ubyte) * SIZE * SIZE;
    printf("%d", boardSize);
    board = (ubyte *)malloc(boardSize);
    zeroWorld(board, SIZE);
    o1(board, SIZE);
    printWorld(board, SIZE);

    ubyte *inboard, *bufferboard;
    cudaMalloc((void **)&inboard, boardSize);
    cudaMalloc((void **)&bufferboard, boardSize);

    cudaMemcpy(inboard, board, boardSize, cudaMemcpyHostToDevice);
    // Time some stuff
    struct timeval t0, t1;
    gettimeofday(&t0, NULL);

    ushort threadsCount = 1024;
    // dim3 threadsPerBlock(SIZE, SIZE);
    // dim3 numBlocks(1);
    assert((SIZE * SIZE) % threadsCount == 0);
    size_t reqBlocksCount = (SIZE * SIZE) / threadsCount;
    ushort blocksCount = (ushort)std::min((size_t)32768, reqBlocksCount);
    printf("blocks %d threads %d", blocksCount, threadsCount);
    int turn, turns = 1000000;
    for (turn = 0; turn < turns; turn++)
    {

        simpleLifeKernel<<<blocksCount, threadsCount>>>(inboard, bufferboard, SIZE);

        std::swap(inboard, bufferboard);
    }
    gettimeofday(&t1, NULL);
    cudaMemcpy(board, inboard, boardSize, cudaMemcpyDeviceToHost);
    printWorld(board, SIZE);

    float seconds = t1.tv_sec - t0.tv_sec + 1E-6 * (t1.tv_usec - t0.tv_usec);
    int worldsize = SIZE * SIZE;
    float MM = (turns * 1.0) * worldsize;
    float MMcellsCalculated = MM / 1000000;
    float total = MMcellsCalculated / seconds;
    printf("Did %f million cells per second in %f\n", total, seconds);

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
