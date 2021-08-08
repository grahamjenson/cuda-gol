package main

import (
	"fmt"
	"runtime"
	"sync"
	"time"
)

// from https://www.fenixara.com/conways-game-of-life-solution-in-golang/

func main() {
	// Size of the world in m x n
	size := 256
	gen := 100000

	ncells := size * size
	cpus := runtime.NumCPU()
	concurrency := cpus
	if size < cpus {
		concurrency = size
	}

	world := make([]int, ncells)
	worldBuffer := make([]int, ncells)

	// Get the initial live cells in the world
	copy([][]int{
		{0, 0, 0, 1, 0},
		{0, 1, 0, 1, 0},
		{0, 0, 1, 1, 0},
		{0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0},
	}, world, size)

	start := time.Now()
	for i := 0; i < gen; i++ {
		computeNextGen(world, worldBuffer, size, concurrency)
		// swap
		tmp := world
		world = worldBuffer
		worldBuffer = tmp
	}
	duration := time.Since(start)

	//printWorld(world, size)
	MMcellsPerSecond := float64((ncells * gen)) / (duration.Seconds() * 1000000)
	fmt.Println("Million Cells Per Second", MMcellsPerSecond, "in", duration.Seconds(), "con", concurrency)
}

func computeNextGen(univ []int, buffer []int, size int, concurrency int) {
	var wg sync.WaitGroup
	wg.Add(concurrency)
	//fromto, univ/2
	for job := 0; job < concurrency; job++ {
		go func(job int) {
			defer func() { wg.Done() }()
			// Chunk the jobs into bits
			x1 := job * (size / concurrency)
			x2 := (job + 1) * (size / concurrency)
			for x := x1; x < x2; x++ {
				for y := 0; y < size; y++ {
					computeCell(x, y, univ, buffer, size)
				}
			}
		}(job)
	}
	wg.Wait()
}

func computeCell(x, y int, inworld []int, buffer []int, size int) {
	//printf("(%d,%d) (%d %d) (%d %d) (%d, %d) (%d, %d)\n", x, y, threadIdx.x, threadIdx.y, blockDim.x, blockDim.y, blockIdx.x, blockIdx.y, gridDim.x, gridDim.y);
	xLeft := ((x + size) - 1) % size
	xRight := (x + 1) % size
	yUp := (y + 1) % size
	yDown := (y + size - 1) % size
	yAbs := size * y
	yUpAbs := yUp * size
	yDownAbs := yDown * size

	Abs := x + yAbs
	aliveCells := inworld[xLeft+yUpAbs] + inworld[x+yUpAbs] + inworld[xRight+yUpAbs] + inworld[xLeft+yAbs] + inworld[xRight+yAbs] + inworld[xLeft+yDownAbs] + inworld[x+yDownAbs] + inworld[xRight+yDownAbs]

	//Any live cell with two or three live neighbours survives.
	//Any dead cell with three live neighbours becomes a live cell.
	//All other live cells die in the next generation. Similarly, all other dead cells stay dead.
	if aliveCells == 3 || (aliveCells == 2 && (inworld[Abs] > 0)) {
		buffer[Abs] = 1
	} else {
		buffer[Abs] = 0
	}

}

func copy(pattern [][]int, world []int, size int) {

	patternsize := len(pattern)

	for y := 0; y < patternsize; y++ {
		for x := 0; x < patternsize; x++ {
			world[x+(size*y)] = pattern[y][x]
		}
	}
}

func printWorld(world []int, size int) {

	fmt.Printf("   ------   \n")
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			fmt.Printf("%d", world[x+(y*size)])
		}
		fmt.Printf("\n")
	}
	fmt.Printf("   ------   \n\n")
}
