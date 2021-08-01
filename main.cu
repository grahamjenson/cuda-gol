#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#define SIZE 32
// Max threadsize is 1024 32*32

void printWorld(short *board);
void zeroWorld(short *board);

__device__ __host__ short coords(short x, short y, short size);

__global__ void playTurn(short *inboard, short *outboard, short size)
{
    short x = threadIdx.x;
    short y = threadIdx.y;

    short val = inboard[coords(x - 1, y - 1, size)] + inboard[coords(x - 1, y, size)] + inboard[coords(x - 1, y + 1, size)] + inboard[coords(x, y - 1, size)] + inboard[coords(x, y + 1, size)] + inboard[coords(x + 1, y - 1, size)] + inboard[coords(x + 1, y, size)] + inboard[coords(x + 1, y + 1, size)];
    bool alive = inboard[coords(x, y, size)] == 1;

    //Any live cell with two or three live neighbours survives.
    //Any dead cell with three live neighbours becomes a live cell.
    //All other live cells die in the next generation. Similarly, all other dead cells stay dead.
    if (alive && (val == 2 || val == 3))
    {
        outboard[coords(x, y, size)] = 1;
    }
    else if (!alive && val == 3)
    {
        outboard[coords(x, y, size)] = 1;
    }
    else
    {
        outboard[coords(x, y, size)] = 0;
    }
}

__device__ __host__ short coords(short x, short y, short size)
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

void o1(short *board)
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
            board[coords(x, y, SIZE)] = o1[y][x];
        }
    }
}

void playTurnD(short *board, short *inboard, short *outboard, short boardSize)
{
    dim3 threadsPerBlock(SIZE, SIZE);
    dim3 numBlocks(1);
    cudaMemcpy(inboard, board, boardSize, cudaMemcpyHostToDevice);
    playTurn<<<numBlocks, threadsPerBlock>>>(inboard, outboard, SIZE);
    cudaMemcpy(board, outboard, boardSize, cudaMemcpyDeviceToHost);
}

int main()
{

    // boardposition = width*heigth cudaboard[width*height] == position
    short *board;
    short boardSize = sizeof(int) * SIZE * SIZE;
    board = (short *)malloc(boardSize);
    zeroWorld(board);
    o1(board);
    printWorld(board);
    // CUDA CALC

    short *inboard, *outboard;
    cudaMalloc((void **)&inboard, boardSize);
    cudaMalloc((void **)&outboard, boardSize);

    // Time some stuff
    struct timeval t0, t1;
    gettimeofday(&t0, NULL);

    int turn, turns = 100000;
    for (turn = 0; turn < turns; turn++)
    {
        playTurnD(board, inboard, outboard, boardSize);
    }
    gettimeofday(&t1, NULL);
    float seconds = t1.tv_sec - t0.tv_sec + 1E-6 * (t1.tv_usec - t0.tv_usec);
    int MMcellsCalculated = (turns * SIZE * SIZE) / 1000000;
    printf("Did 1 calls in %f seconds\n", MMcellsCalculated / seconds);

    cudaFree(inboard);
    cudaFree(outboard);
    free(board);

    return 0;
}

void zeroWorld(short *board)
{
    short x, y;

    for (y = 0; y < SIZE; ++y)
    {
        for (x = 0; x < SIZE; ++x)
        {
            board[coords(x, y, SIZE)] = 0;
        }
    }
}

void printWorld(short *board)
{
    short x, y;

    printf("   ------   \n");
    for (y = 0; y < SIZE; y++)
    {
        for (x = 0; x < SIZE; x++)
        {
            printf("%d", board[coords(x, y, SIZE)]);
        }
        printf("\n");
    }
    printf("   ------   \n\n");
}
