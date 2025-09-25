package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"

	"github.com/golang/geo/r3"
	"go.viam.com/rdk/spatialmath"
)

// PoseData stores raw pose information from Viam
type PoseData struct {
	Pose  int     `json:"pose"`
	X     float64 `json:"x"`
	Y     float64 `json:"y"`
	Z     float64 `json:"z"`
	OX    float64 `json:"ox"`
	OY    float64 `json:"oy"`
	OZ    float64 `json:"oz"`
	Theta float64 `json:"theta"`
}

// RawCalibrationData stores raw pose data from both arm and AprilTag
type RawCalibrationData struct {
	ArmPosesJson []PoseData `json:"arm_poses"`
	TagPosesJson []PoseData `json:"tag_poses"`
}

// CalibrationData matches the format expected by OpenCV hand-eye calibration
type CalibrationData struct {
	RGripper2Base [][]float64 `json:"R_gripper2base"`
	TGripper2Base [][]float64 `json:"t_gripper2base"`
	RTarget2Cam   [][]float64 `json:"R_target2cam"`
	TTarget2Cam   [][]float64 `json:"t_target2cam"`
}

func other() {
	// Read raw calibration data from JSON file
	jsonData, err := os.ReadFile("raw_calibration_data.json")
	if err != nil {
		log.Fatalf("Failed to read calibration data: %v", err)
	}

	var calibrationData RawCalibrationData
	err = json.Unmarshal(jsonData, &calibrationData)
	if err != nil {
		log.Fatalf("Failed to unmarshal calibration data: %v", err)
	}

	fmt.Printf("Loaded raw calibration data from raw_calibration_data.json\n")
	fmt.Printf("Found %d arm poses and %d tag poses\n", len(calibrationData.ArmPosesJson), len(calibrationData.TagPosesJson))

	// Print out the data
	for i, armPoseJson := range calibrationData.ArmPosesJson {
		fmt.Printf("\n--- Pose %d ---\n", i+1)
		fmt.Printf("Arm Position (x,y,z): (%.3f, %.3f, %.3f) mm\n",
			armPoseJson.X, armPoseJson.Y, armPoseJson.Z)
		fmt.Printf("Arm Orientation (OX,OY,OZ,Theta): (%.3f, %.3f, %.3f, %.3f)\n",
			armPoseJson.OX, armPoseJson.OY, armPoseJson.OZ, armPoseJson.Theta)

		armPose := spatialmath.NewPose(
			r3.Vector{X: armPoseJson.X, Y: armPoseJson.Y, Z: armPoseJson.Z},
			&spatialmath.OrientationVectorDegrees{OX: armPoseJson.OX, OY: armPoseJson.OY, OZ: armPoseJson.OZ, Theta: armPoseJson.Theta},
		)
		fmt.Printf("arm pose: %v\n", armPose)
		fmt.Printf("arm pose quaternion: %v\n", armPose.Orientation().Quaternion())
		fmt.Printf("arm pose rotation matrix: %v\n", armPose.Orientation().RotationMatrix())

		if i < len(calibrationData.TagPosesJson) {
			tagPoseJson := calibrationData.TagPosesJson[i]
			fmt.Printf("Tag Position (x,y,z): (%.3f, %.3f, %.3f) mm\n",
				tagPoseJson.X, tagPoseJson.Y, tagPoseJson.Z)
			fmt.Printf("Tag Orientation (OX,OY,OZ,Theta): (%.3f, %.3f, %.3f, %.3f)\n",
				tagPoseJson.OX, tagPoseJson.OY, tagPoseJson.OZ, tagPoseJson.Theta)

			tagPose := spatialmath.NewPose(
				r3.Vector{X: tagPoseJson.X, Y: tagPoseJson.Y, Z: tagPoseJson.Z},
				&spatialmath.OrientationVectorDegrees{OX: tagPoseJson.OX, OY: tagPoseJson.OY, OZ: tagPoseJson.OZ, Theta: tagPoseJson.Theta},
			)
			fmt.Printf("Tag pose: %v\n", tagPose)
			fmt.Printf("Tag pose quaternion: %v\n", tagPose.Orientation().Quaternion())
			fmt.Printf("Tag pose rotation matrix: %v\n", tagPose.Orientation().RotationMatrix())
		}
	}

	// Convert to OpenCV calibration format
	fmt.Printf("\n--- Converting to OpenCV calibration format ---\n")

	opencvData := CalibrationData{
		RGripper2Base: [][]float64{},
		TGripper2Base: [][]float64{},
		RTarget2Cam:   [][]float64{},
		TTarget2Cam:   [][]float64{},
	}

	// Process each pose pair
	for i := 0; i < len(calibrationData.ArmPosesJson) && i < len(calibrationData.TagPosesJson); i++ {
		// Convert arm pose to Viam Pose
		armPoseJson := calibrationData.ArmPosesJson[i]
		armPose := spatialmath.NewPose(
			r3.Vector{X: armPoseJson.X, Y: armPoseJson.Y, Z: armPoseJson.Z},
			&spatialmath.OrientationVectorDegrees{OX: armPoseJson.OX, OY: armPoseJson.OY, OZ: armPoseJson.OZ, Theta: armPoseJson.Theta},
		)

		// Convert tag pose to Viam Pose
		tagPoseJson := calibrationData.TagPosesJson[i]
		tagPose := spatialmath.NewPose(
			r3.Vector{X: tagPoseJson.X, Y: tagPoseJson.Y, Z: tagPoseJson.Z},
			&spatialmath.OrientationVectorDegrees{OX: tagPoseJson.OX, OY: tagPoseJson.OY, OZ: tagPoseJson.OZ, Theta: tagPoseJson.Theta},
		)

		// Convert to rotation matrices and translation vectors
		armR := rotationMatrixToSlice(*armPose.Orientation().RotationMatrix())
		armT := []float64{armPose.Point().X, armPose.Point().Y, armPose.Point().Z}

		tagR := rotationMatrixToSlice(*tagPose.Orientation().RotationMatrix())
		tagT := []float64{tagPose.Point().X, tagPose.Point().Y, tagPose.Point().Z}

		// Add to calibration data
		opencvData.RGripper2Base = append(opencvData.RGripper2Base, armR)
		opencvData.TGripper2Base = append(opencvData.TGripper2Base, armT)
		opencvData.RTarget2Cam = append(opencvData.RTarget2Cam, tagR)
		opencvData.TTarget2Cam = append(opencvData.TTarget2Cam, tagT)
	}

	// Save to JSON file
	jsonData, err = json.MarshalIndent(opencvData, "", "  ")
	if err != nil {
		log.Fatalf("Failed to marshal OpenCV calibration data: %v", err)
	}

	err = os.WriteFile("go_calibration.json", jsonData, 0644)
	if err != nil {
		log.Fatalf("Failed to write OpenCV calibration data: %v", err)
	}

	fmt.Printf("OpenCV calibration data saved to go_calibration.json\n")
	fmt.Printf("Generated %d pose pairs for hand-eye calibration\n", len(opencvData.RGripper2Base))
}

// rotationMatrixToSlice converts a rotation matrix to a flattened slice for JSON serialization
func rotationMatrixToSlice(rotMat spatialmath.RotationMatrix) []float64 {
	return []float64{
		rotMat.At(0, 0), rotMat.At(0, 1), rotMat.At(0, 2),
		rotMat.At(1, 0), rotMat.At(1, 1), rotMat.At(1, 2),
		rotMat.At(2, 0), rotMat.At(2, 1), rotMat.At(2, 2),
	}
}
