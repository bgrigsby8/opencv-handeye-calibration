package main

import (
	"fmt"
	"os"
	"strconv"

	"github.com/golang/geo/r3"
	"go.viam.com/rdk/spatialmath"
)

func main() {
	// Check if we have the correct number of arguments
	if len(os.Args) != 5 {
		fmt.Fprintf(os.Stderr, "Usage: %s <ox> <oy> <oz> <theta>\n", os.Args[0])
		os.Exit(1)
	}

	// Parse command line arguments
	ox, err := strconv.ParseFloat(os.Args[1], 64)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error parsing ox: %v\n", err)
		os.Exit(1)
	}

	oy, err := strconv.ParseFloat(os.Args[2], 64)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error parsing oy: %v\n", err)
		os.Exit(1)
	}

	oz, err := strconv.ParseFloat(os.Args[3], 64)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error parsing oz: %v\n", err)
		os.Exit(1)
	}

	theta, err := strconv.ParseFloat(os.Args[4], 64)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error parsing theta: %v\n", err)
		os.Exit(1)
	}

	// Create orientation vector from the input parameters
	orientationVector := &spatialmath.OrientationVectorDegrees{
		OX:    ox,
		OY:    oy,
		OZ:    oz,
		Theta: theta,
	}

	// Create a pose with zero translation and the given orientation
	pose := spatialmath.NewPose(
		r3.Vector{X: 0, Y: 0, Z: 0}, // Zero translation
		orientationVector,
	)

	// Get the rotation matrix
	rotMatrix := pose.Orientation().RotationMatrix()

	// Output the rotation matrix as 9 space-separated values (row-major order)
	// This will be parsed by the Python script
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			fmt.Printf("%.10f ", rotMatrix.At(i, j))
		}
	}
	fmt.Println() // Final newline
}