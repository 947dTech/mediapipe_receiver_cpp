#pragma once

#ifndef MEDIAPIPE_RECEIVER_CPP_FUNCTIONS_H_
#define MEDIAPIPE_RECEIVER_CPP_FUNCTIONS_H_

#include <vector>
#include <Eigen/Geometry>
#include <Eigen/Dense>

namespace mediapipe_receiver {

const size_t kNPoseLandmarks = 33;
const size_t kNFaceLandmarks = 468;
const size_t kNHandLandmarks = 21;

enum class CoordinatesSet {
	FBX,
	VRM
};

/// @brief make 3x3 rotation matrix using 3 vectors
/// @param[out] co target matrix
/// @param[in] x x vector
/// @param[in] y y vector
/// @param[in] z z vector
inline void MakeMatrixFromVec(
	Eigen::Matrix3f& co,
	const Eigen::Vector3f& x,
	const Eigen::Vector3f& y,
	const Eigen::Vector3f& z)
{
	for (size_t i = 0; i < 3; i++) {
		co(i, 0) = x[i];
		co(i, 1) = y[i];
		co(i, 2) = z[i];
	}
}

/// @brief transform all points using trans / rot
/// @param points target points, all points are overwritten by transformed points
/// @param[in] tf_trans translation vector
/// @param[in] tf_rot rotation matrix
inline void TransformPoints(
	std::vector	<Eigen::Vector3f>& points,
	const Eigen::Vector3f& tf_trans,
	const Eigen::Matrix3f& tf_rot)
{
	for (size_t i = 0; i < points.size(); i++) {
		Eigen::Vector3f& point = points[i];
		point = tf_rot * point + tf_trans;
	}
}

/// @brief convert point from camera coordinate (right hand) to model coordinate (left hand)
/// @param camera_pos
/// @param model_pos
inline void CameraToModel(const Eigen::Vector3f& camera_pos, Eigen::Vector3f& model_pos)
{
	// swap axis into model coordinate (left hand)
	//model_pos[0] = camera_pos[0];
	//model_pos[1] = -camera_pos[2];
	model_pos[0] = -camera_pos[0];
	model_pos[1] = camera_pos[2];
	model_pos[2] = 1.0 - camera_pos[1];
}

}  // namespace mediapipe_receiver

#endif  // MEDIAPIPE_RECEIVER_CPP_FUNCTIONS_H_
