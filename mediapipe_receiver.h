#pragma once

#ifndef MEDIAPIPE_RECEIVER_CPP_MEDIAPIPE_RECEIVER_H_
#define MEDIAPIPE_RECEIVER_CPP_MEDIAPIPE_RECEIVER_H_

#include <vector>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <cmath>

#include "functions.h"
#include "face.h"
#include "hand.h"

namespace mediapipe_receiver {

class MediapipeReceiver {
private:
	// constant
	float pi_;

	// camera params
	float focal_length_;
	float frame_width_;
	float frame_height_;
	float aspect_ratio_;

	// gravity
	std::vector< float > gravity_;
	Eigen::Vector3f camera_gravity_;
	Eigen::Vector3f camera_gravity_pos_;
	Eigen::Matrix3f camera_gravity_co_;

	Eigen::Vector3f inv_gravity_trans_;
	Eigen::Matrix3f inv_gravity_rot_;

	// ladmarks
	// NOTE: There are 3 types of coordinate;
	// - <part>_points_ : in the sensor coordinate
	// - camera_<part>_points_ : in the gravity-aligned camera coordinate (right-hand)
	// - model_<part>_points_ : in the model coordinate (left-hand)

	// pose
	std::vector < std::vector< float > > pose_points_;
	std::vector	<Eigen::Vector3f> camera_pose_points_;
	std::vector	<Eigen::Vector3f> model_pose_points_;

	// pose_world
	std::vector < std::vector< float > > pose_2d_points_;
	std::vector	<Eigen::Vector3f> camera_pose_2d_points_;
	std::vector	<Eigen::Vector3f> model_pose_2d_points_;

	// face
	std::vector < std::vector< float > > face_points_;
	std::vector	<Eigen::Vector3f> camera_face_points_;
	std::vector	<Eigen::Vector3f> model_face_points_;

	// right_hand
	std::vector < std::vector< float > > right_hand_points_;
	std::vector	<Eigen::Vector3f> camera_right_hand_points_;
	std::vector	<Eigen::Vector3f> model_right_hand_points_;

	// left_hand
	std::vector < std::vector< float > > left_hand_points_;
	std::vector	<Eigen::Vector3f> camera_left_hand_points_;
	std::vector	<Eigen::Vector3f> model_left_hand_points_;

	// coordinates
	// base poses (global)
	Eigen::Matrix3f pelvis_base_co_;  // as ROOT
	Eigen::Matrix3f shoulder_base_co_;

	Eigen::Matrix3f upper_arm_r_base_co_;
	Eigen::Matrix3f elbow_r_base_co_;
	Eigen::Matrix3f phand_r_base_co_;

	Eigen::Matrix3f upper_arm_l_base_co_;
	Eigen::Matrix3f elbow_l_base_co_;
	Eigen::Matrix3f phand_l_base_co_;

	Eigen::Matrix3f thigh_r_base_co_;
	Eigen::Matrix3f knee_r_base_co_;
	Eigen::Matrix3f ankle_r_base_co_;

	Eigen::Matrix3f thigh_l_base_co_;
	Eigen::Matrix3f knee_l_base_co_;
	Eigen::Matrix3f ankle_l_base_co_;

	// base poses (local)
	Eigen::Matrix3f local_shoulder_base_co_;

	Eigen::Matrix3f local_upper_arm_r_base_co_;
	Eigen::Matrix3f local_elbow_r_base_co_;
	Eigen::Matrix3f local_phand_r_base_co_;

	Eigen::Matrix3f local_upper_arm_l_base_co_;
	Eigen::Matrix3f local_elbow_l_base_co_;
	Eigen::Matrix3f local_phand_l_base_co_;

	Eigen::Matrix3f local_thigh_r_base_co_;
	Eigen::Matrix3f local_knee_r_base_co_;
	Eigen::Matrix3f local_ankle_r_base_co_;

	Eigen::Matrix3f local_thigh_l_base_co_;
	Eigen::Matrix3f local_knee_l_base_co_;
	Eigen::Matrix3f local_ankle_l_base_co_;

	// current poses (global)
	Eigen::Matrix3f pelvis_co_;
	Eigen::Matrix3f shoulder_co_;
	Eigen::Matrix3f shoulder_ub_co_;

	Eigen::Matrix3f upper_arm_r_co_;
	Eigen::Matrix3f elbow_r_co_;
	Eigen::Matrix3f phand_r_co_;

	Eigen::Matrix3f upper_arm_l_co_;
	Eigen::Matrix3f elbow_l_co_;
	Eigen::Matrix3f phand_l_co_;

	Eigen::Matrix3f thigh_r_co_;
	Eigen::Matrix3f knee_r_co_;
	Eigen::Matrix3f ankle_r_co_;

	Eigen::Matrix3f thigh_l_co_;
	Eigen::Matrix3f knee_l_co_;
	Eigen::Matrix3f ankle_l_co_;

	// diff to local base coords
	Eigen::Matrix3f shoulder_rot_;
	Eigen::Matrix3f shoulder_ub_rot_;

	Eigen::Matrix3f upper_arm_l_rot_;
	Eigen::Matrix3f elbow_l_rot_;
	Eigen::Matrix3f phand_l_rot_;

	Eigen::Matrix3f upper_arm_r_rot_;
	Eigen::Matrix3f elbow_r_rot_;
	Eigen::Matrix3f phand_r_rot_;

	Eigen::Matrix3f pelvis_rot_;

	Eigen::Matrix3f thigh_l_rot_;
	Eigen::Matrix3f knee_l_rot_;
	Eigen::Matrix3f ankle_l_rot_;

	Eigen::Matrix3f thigh_r_rot_;
	Eigen::Matrix3f knee_r_rot_;
	Eigen::Matrix3f ankle_r_rot_;

	// offset
	float arm_adjust_angle_;
	float pelvis_adjust_angle_;
	float ankle_adjust_angle_;
	float upper_arm_roll_offset_;
	float elbow_offset_;
	float thigh_roll_offset_;
	float knee_offset_;
	float neck_pitch_offset_;

	// face
	FaceParams face_params_;
	Eigen::Matrix3f face_co_;
	Eigen::Matrix3f face_rot_;
	Eigen::Matrix3f face_ub_rot_;

	// hand
	HandParams right_hand_params_;
	HandParams left_hand_params_;

	// switch whether FBX or VRM
	CoordinatesSet mode_;

public:
	MediapipeReceiver()
	{
		pi_ = acos(-1.0);
		float focal_length = 872.6;  // xperia 1 iv
		// float focal_length = 895.5;  // pixel 5
		float frame_width = 1280.0;
		float frame_height = 720.0;
		SetCameraParams(focal_length, frame_width, frame_height);

		gravity_.resize(3);

		camera_gravity_co_ <<
			0.0, -1.0, 0.0,
			-1.0, 0.0, 0.0,
			0.0, 0.0, -1.0;

		// reserve all point buffers
		pose_points_.resize(kNPoseLandmarks);
		for (size_t i = 0; i < kNPoseLandmarks; i++) {
			pose_points_[i].resize(5);
		}
		camera_pose_points_.resize(kNPoseLandmarks);
		model_pose_points_.resize(kNPoseLandmarks);

		pose_2d_points_.resize(kNPoseLandmarks);
		for (size_t i = 0; i < kNPoseLandmarks; i++) {
			pose_2d_points_[i].resize(5);
		}
		camera_pose_2d_points_.resize(kNPoseLandmarks);
		model_pose_2d_points_.resize(kNPoseLandmarks);

		face_points_.resize(kNFaceLandmarks);
		for (size_t i = 0; i < kNFaceLandmarks; i++) {
			face_points_[i].resize(5);
		}
		camera_face_points_.resize(kNFaceLandmarks);
		model_face_points_.resize(kNFaceLandmarks);

		right_hand_points_.resize(kNHandLandmarks);
		for (size_t i = 0; i < kNHandLandmarks; i++) {
			right_hand_points_[i].resize(5);
		}
		camera_right_hand_points_.resize(kNHandLandmarks);
		model_right_hand_points_.resize(kNHandLandmarks);

		left_hand_points_.resize(kNHandLandmarks);
		for (size_t i = 0; i < kNHandLandmarks; i++) {
			left_hand_points_[i].resize(5);
		}
		camera_left_hand_points_.resize(kNHandLandmarks);
		model_left_hand_points_.resize(kNHandLandmarks);

		// NOTE: please replace this adjust angle according to zero pose of your model
		// if model is A-pose, set adjust angle = (arm angle from spine), ex. 50.0
		// if model is T-pose, set adjust angle = 90.0
		arm_adjust_angle_ = 90.0;
		// NOTE: for VRM, adjust_angle_ = 0.0
		pelvis_adjust_angle_ = -15.0;
		ankle_adjust_angle_ = 150.0;

		InitPoses();

		right_hand_params_.SetRightHand();
		left_hand_params_.SetLeftHand();

		upper_arm_roll_offset_ = 0.0;
		elbow_offset_ = 0.0;
		thigh_roll_offset_ = 0.0;
		knee_offset_ = 0.0;
		neck_pitch_offset_ = 0.0;
	}

	~MediapipeReceiver()
	{
	}

	float Degrees(float rad)
	{
		return rad / pi_ * 180.0;
	}

	float Radians(float deg)
	{
		return deg * pi_ / 180.0;
	}

	void arm_adjust_angle(float val)
	{
		arm_adjust_angle_ = val;
	}

	float arm_adjust_angle()
	{
		return arm_adjust_angle_;
	}

	void pelvis_adjust_angle(float val)
	{
		pelvis_adjust_angle_ = val;
	}

	float pelvis_adjust_angle()
	{
		return pelvis_adjust_angle_;
	}

	void ankle_adjust_angle(float val)
	{
		ankle_adjust_angle_ = val;
	}

	float ankle_adjust_angle()
	{
		return ankle_adjust_angle_;
	}

	void InitPoses(CoordinatesSet mode = CoordinatesSet::FBX)
	{
		mode_ = mode;
		right_hand_params_.SetMode(mode_);
		left_hand_params_.SetMode(mode_);

		// NOTE: eigen is column-major, right hand coordinate system
		// set pelvis as ROOT
		if (mode_ == CoordinatesSet::FBX) {
			pelvis_base_co_ =
				Eigen::AngleAxisf(-0.5 * pi_, Eigen::Vector3f::UnitX())
				* Eigen::AngleAxisf(Radians(pelvis_adjust_angle_), Eigen::Vector3f::UnitX());
		} else if (mode_ == CoordinatesSet::VRM) {
			pelvis_base_co_ = Eigen::Matrix3f::Identity();
		}

		// global poses
		// NOTE: shoulder_adjust_angle_ must be set independently?
		shoulder_base_co_ = pelvis_base_co_;
		//shoulder_base_co_ = Eigen::AngleAxisf(-0.5 * pi_, Eigen::Vector3f::UnitX());

		if (mode_ == CoordinatesSet::FBX) {
			upper_arm_l_base_co_ =
				Eigen::AngleAxisf(-0.5 * pi_, Eigen::Vector3f::UnitX())
				* Eigen::AngleAxisf(Radians(180.0 - arm_adjust_angle_), Eigen::Vector3f::UnitZ());
		} else if (mode_ == CoordinatesSet::VRM) {
			upper_arm_l_base_co_ = Eigen::Matrix3f::Identity();
		}

		elbow_l_base_co_ = upper_arm_l_base_co_;
		phand_l_base_co_ = upper_arm_l_base_co_;

		if (mode_ == CoordinatesSet::FBX) {
			upper_arm_r_base_co_ =
				Eigen::AngleAxisf(-0.5 * pi_, Eigen::Vector3f::UnitX())
				* Eigen::AngleAxisf(Radians(180.0 + arm_adjust_angle_), Eigen::Vector3f::UnitZ());
		} else if (mode_ == CoordinatesSet::VRM) {
			upper_arm_r_base_co_ = Eigen::Matrix3f::Identity();
		}
		elbow_r_base_co_ = upper_arm_r_base_co_;
		phand_r_base_co_ = upper_arm_r_base_co_;

		if (mode_ == CoordinatesSet::FBX) {
			//thigh_l_base_co_ =
			//	Eigen::AngleAxisf(-0.5 * pi_, Eigen::Vector3f::UnitX())
			//	* Eigen::AngleAxisf(pi_, Eigen::Vector3f::UnitZ());
			thigh_l_base_co_ =
				Eigen::AngleAxisf(0.5 * pi_, Eigen::Vector3f::UnitX());
			knee_l_base_co_ = thigh_l_base_co_;
			//ankle_l_base_co_ = thigh_l_base_co_;
			ankle_l_base_co_ =
				Eigen::AngleAxisf(Radians(ankle_adjust_angle_), Eigen::Vector3f::UnitX());
		} else if (mode_ == CoordinatesSet::VRM) {
			thigh_l_base_co_ = Eigen::Matrix3f::Identity();
			knee_l_base_co_ = thigh_l_base_co_;
			ankle_l_base_co_ = thigh_l_base_co_;
		}

		if (mode_ == CoordinatesSet::FBX) {
			//thigh_r_base_co_ =
			//	Eigen::AngleAxisf(-0.5 * pi_, Eigen::Vector3f::UnitX())
			//	* Eigen::AngleAxisf(pi_, Eigen::Vector3f::UnitZ());
			thigh_r_base_co_ =
				Eigen::AngleAxisf(0.5 * pi_, Eigen::Vector3f::UnitX());
			knee_r_base_co_ = thigh_r_base_co_;
			//ankle_r_base_co_ = thigh_r_base_co_;
			ankle_r_base_co_ =
				Eigen::AngleAxisf(Radians(ankle_adjust_angle_), Eigen::Vector3f::UnitX());
		} else if (mode_ == CoordinatesSet::VRM) {
			thigh_r_base_co_ = Eigen::Matrix3f::Identity();
			knee_r_base_co_ = thigh_r_base_co_;
			ankle_r_base_co_ = thigh_r_base_co_;
		}

		// local poses
		local_shoulder_base_co_ = pelvis_base_co_.transpose() * shoulder_base_co_;

		local_upper_arm_l_base_co_ = shoulder_base_co_.transpose() * upper_arm_l_base_co_;
		local_elbow_l_base_co_ = upper_arm_l_base_co_.transpose() * elbow_l_base_co_;
		local_phand_l_base_co_ = elbow_l_base_co_.transpose() * phand_l_base_co_;

		local_upper_arm_r_base_co_ = shoulder_base_co_.transpose() * upper_arm_r_base_co_;
		local_elbow_r_base_co_ = upper_arm_r_base_co_.transpose() * elbow_r_base_co_;
		local_phand_r_base_co_ = elbow_r_base_co_.transpose() * phand_r_base_co_;

		local_thigh_l_base_co_ = pelvis_base_co_.transpose() * thigh_l_base_co_;
		local_knee_l_base_co_ = thigh_l_base_co_.transpose() * knee_l_base_co_;
		local_ankle_l_base_co_ = knee_l_base_co_.transpose() * ankle_l_base_co_;

		local_thigh_r_base_co_ = pelvis_base_co_.transpose() * thigh_r_base_co_;
		local_knee_r_base_co_ = thigh_r_base_co_.transpose() * knee_r_base_co_;
		local_ankle_r_base_co_ = knee_r_base_co_.transpose() * ankle_r_base_co_;
	}

	Eigen::Matrix3f& pelvis_co()
	{
		return pelvis_co_;
	}

	Eigen::Matrix3f& shoulder_co()
	{
		return shoulder_co_;
	}

	// tmporary
	Eigen::Matrix3f& local_shoulder_base_co()
	{
		return local_shoulder_base_co_;
	}

	Eigen::Matrix3f& shoulder_ub_co()
	{
		return shoulder_ub_co_;
	}

	Eigen::Matrix3f& upper_arm_r_co()
	{
		return upper_arm_r_co_;
	}

	Eigen::Matrix3f& elbow_r_co()
	{
		return elbow_r_co_;
	}

	Eigen::Matrix3f& phand_r_co()
	{
		return phand_r_co_;
	}

	Eigen::Matrix3f& upper_arm_l_co()
	{
		return upper_arm_l_co_;
	}

	Eigen::Matrix3f& elbow_l_co()
	{
		return elbow_l_co_;
	}

	Eigen::Matrix3f& phand_l_co()
	{
		return phand_l_co_;
	}

	Eigen::Matrix3f& thigh_r_co()
	{
		return thigh_r_co_;
	}

	Eigen::Matrix3f& knee_r_co()
	{
		return knee_r_co_;
	}

	Eigen::Matrix3f& ankle_r_co()
	{
		return ankle_r_co_;
	}

	Eigen::Matrix3f& thigh_l_co()
	{
		return thigh_l_co_;
	}

	Eigen::Matrix3f& knee_l_co()
	{
		return knee_l_co_;
	}

	Eigen::Matrix3f& ankle_l_co()
	{
		return ankle_l_co_;
	}

	Eigen::Matrix3f& shoulder_rot()
	{
		return shoulder_rot_;
	}

	Eigen::Matrix3f& shoulder_ub_rot()
	{
		return shoulder_ub_rot_;
	}

	Eigen::Matrix3f& upper_arm_l_rot()
	{
		return upper_arm_l_rot_;
	}

	Eigen::Matrix3f& elbow_l_rot()
	{
		return elbow_l_rot_;
	}

	Eigen::Matrix3f& phand_l_rot()
	{
		return phand_l_rot_;
	}

	Eigen::Matrix3f& upper_arm_r_rot()
	{
		return upper_arm_r_rot_;
	}

	Eigen::Matrix3f& elbow_r_rot()
	{
		return elbow_r_rot_;
	}

	Eigen::Matrix3f& phand_r_rot()
	{
		return phand_r_rot_;
	}

	Eigen::Matrix3f& pelvis_rot()
	{
		return pelvis_rot_;
	}

	Eigen::Matrix3f& thigh_l_rot()
	{
		return thigh_l_rot_;
	}

	Eigen::Matrix3f& knee_l_rot()
	{
		return knee_l_rot_;
	}

	Eigen::Matrix3f& ankle_l_rot()
	{
		return ankle_l_rot_;
	}

	Eigen::Matrix3f& thigh_r_rot()
	{
		return thigh_r_rot_;
	}

	Eigen::Matrix3f& knee_r_rot()
	{
		return knee_r_rot_;
	}

	Eigen::Matrix3f& ankle_r_rot()
	{
		return ankle_r_rot_;
	}

	// pose_world
	std::vector < std::vector< float > >& pose_points()
	{
		return pose_points_;
	}

	std::vector	<Eigen::Vector3f>& camera_pose_points()
	{
		return camera_pose_points_;
	}

	std::vector	<Eigen::Vector3f>& model_pose_points()
	{
		return model_pose_points_;
	}

	void TransformPosePointsToDevice()
	{
		camera_pose_points_.resize(pose_points_.size());
		for (size_t i = 0; i < pose_points_.size(); i++) {
			// use Eigen
			std::vector<float>& pos = pose_points_[i];
			Eigen::Vector3f& eigen_pose_point = camera_pose_points_[i];
			// NOTE: 3d landmarks shall be copied
			for (size_t j = 0; j < 3; j++) {
				// NOT swap axis, in camera coordinate (right hand)
				eigen_pose_point[j] = pos[j];
			}
		}

		if (camera_pose_points_.size() == 33) {
			camera_gravity_pos_ = (camera_pose_points_[11] + camera_pose_points_[12]) * 0.5;
		}
		else {
			camera_gravity_pos_ = Eigen::Vector3f::Zero();
		}
	}

	void TransformPosePointsToCamera()
	{
		TransformPoints(camera_pose_points_, inv_gravity_trans_, inv_gravity_rot_);
	}

	void TransformPosePointsToModel()
	{
		model_pose_points_.resize(camera_pose_points_.size());
		for (size_t i = 0; i < camera_pose_points_.size(); i++) {
			CameraToModel(camera_pose_points_[i], model_pose_points_[i]);
		}
	}

	// pose
	std::vector < std::vector< float > >& pose_2d_points()
	{
		return pose_2d_points_;
	}

	std::vector	<Eigen::Vector3f>& camera_pose_2d_points()
	{
		return camera_pose_2d_points_;
	}

	std::vector	<Eigen::Vector3f>& model_pose_2d_points()
	{
		return model_pose_2d_points_;
	}

	// face
	std::vector < std::vector< float > >& face_points()
	{
		return face_points_;
	}

	std::vector	<Eigen::Vector3f>& camera_face_points()
	{
		return camera_face_points_;
	}

	std::vector	<Eigen::Vector3f>& model_face_points()
	{
		return model_face_points_;
	}

	void TransformFacePointsToDevice()
	{
		camera_face_points_.resize(face_points_.size());
		for (size_t i = 0; i < face_points_.size(); i++) {
			// use Eigen
			std::vector<float>& pos = face_points_[i];
			Eigen::Vector3f& eigen_face_point = camera_face_points_[i];
			// NOTE: 2d landmarks shall be converted into uv geometories
			eigen_face_point[0] = (pos[0] - 0.5) / aspect_ratio_;
			eigen_face_point[1] = (pos[1] - 0.5);
			eigen_face_point[2] = (pos[2] - 0.5) / aspect_ratio_;
		}
	}

	void TransformFacePointsToCamera()
	{
		TransformPoints(camera_face_points_, inv_gravity_trans_, inv_gravity_rot_);
	}

	void TransformFacePointsToModel()
	{
		model_face_points_.resize(camera_face_points_.size());
		for (size_t i = 0; i < camera_face_points_.size(); i++) {
			CameraToModel(camera_face_points_[i], model_face_points_[i]);
		}
	}

	// right_hand
	std::vector < std::vector< float > >& right_hand_points()
	{
		return right_hand_points_;
	}

	std::vector	<Eigen::Vector3f>& camera_right_hand_points()
	{
		return camera_right_hand_points_;
	}

	std::vector	<Eigen::Vector3f>& model_right_hand_points()
	{
		return model_right_hand_points_;
	}

	void TransformRightHandPointsToDevice()
	{
		camera_right_hand_points_.resize(right_hand_points_.size());
		for (size_t i = 0; i < right_hand_points_.size(); i++) {
			// use Eigen
			std::vector<float>& pos = right_hand_points_[i];
			Eigen::Vector3f& eigen_right_hand_point = camera_right_hand_points_[i];
			// NOTE: 2d landmarks shall be converted into uv geometories
			eigen_right_hand_point[0] = (pos[0] - 0.5) / aspect_ratio_;
			eigen_right_hand_point[1] = (pos[1] - 0.5);
			eigen_right_hand_point[2] = (pos[2] - 0.5) / aspect_ratio_;
		}
	}

	void TransformRightHandPointsToCamera()
	{
		TransformPoints(camera_right_hand_points_, inv_gravity_trans_, inv_gravity_rot_);
	}

	void TransformRightHandPointsToModel()
	{
		model_right_hand_points_.resize(camera_right_hand_points_.size());
		for (size_t i = 0; i < camera_right_hand_points_.size(); i++) {
			CameraToModel(camera_right_hand_points_[i], model_right_hand_points_[i]);
		}
	}

	// left_hand
	std::vector < std::vector< float > >& left_hand_points()
	{
		return left_hand_points_;
	}

	std::vector	<Eigen::Vector3f>& camera_left_hand_points()
	{
		return camera_left_hand_points_;
	}

	std::vector	<Eigen::Vector3f>& model_left_hand_points()
	{
		return model_left_hand_points_;
	}

	void TransformLeftHandPointsToDevice()
	{
		camera_left_hand_points_.resize(left_hand_points_.size());
		for (size_t i = 0; i < left_hand_points_.size(); i++) {
			// use Eigen
			std::vector<float>& pos = left_hand_points_[i];
			Eigen::Vector3f& eigen_left_hand_point = camera_left_hand_points_[i];
			// NOTE: 2d landmarks shall be converted into uv geometories
			eigen_left_hand_point[0] = (pos[0] - 0.5) / aspect_ratio_;
			eigen_left_hand_point[1] = (pos[1] - 0.5);
			eigen_left_hand_point[2] = (pos[2] - 0.5) / aspect_ratio_;
		}
	}

	void TransformLeftHandPointsToCamera()
	{
		TransformPoints(camera_left_hand_points_, inv_gravity_trans_, inv_gravity_rot_);
	}

	void TransformLeftHandPointsToModel()
	{
		model_left_hand_points_.resize(camera_left_hand_points_.size());
		for (size_t i = 0; i < camera_left_hand_points_.size(); i++) {
			CameraToModel(camera_left_hand_points_[i], model_left_hand_points_[i]);
		}
	}

	// camera params
	float focal_length()
	{
		return focal_length_;
	}

	float frame_width()
	{
		return frame_width_;
	}

	float frame_height()
	{
		return frame_height_;
	}

	float aspect_ratio()
	{
		return aspect_ratio_;
	}

	void SetCameraParams(float focal_length, float frame_width, float frame_height)
	{
		focal_length_ = focal_length;
		frame_width_ = frame_width;
		frame_height_ = frame_height;
		aspect_ratio_ = frame_width_ / frame_height_;
	}

	// gravity
	std::vector<float>& gravity()
	{
		return gravity_;
	}

	Eigen::Vector3f& camera_gravity()
	{
		return camera_gravity_;
	}

	Eigen::Vector3f& camera_gravity_pos()
	{
		return camera_gravity_pos_;
	}

	Eigen::Matrix3f& camera_gravity_co()
	{
		return camera_gravity_co_;
	}

	bool SetGravity(float x, float y, float z)
	{
		bool result = false;

		gravity_[0] = x;
		gravity_[1] = y;
		gravity_[2] = z;

		camera_gravity_[0] = gravity_[0];
		camera_gravity_[1] = -gravity_[1];
		camera_gravity_[2] = -gravity_[2];

		float ngrav = camera_gravity_.norm();
		if (ngrav > 1e-5) {
			camera_gravity_ /= ngrav;

			float slant = fabs(camera_gravity_[2]);
			if (slant < 0.6) {
				Eigen::Vector3f gravity_y = -camera_gravity_;
				Eigen::Vector3f gravity_x = camera_gravity_.cross(Eigen::Vector3f::UnitZ());
				gravity_x /= gravity_x.norm();
				Eigen::Vector3f gravity_z = gravity_x.cross(gravity_y);
				MakeMatrixFromVec(camera_gravity_co_, gravity_x, gravity_y, gravity_z);
				result = true;
			}
		}
		return result;
	}

	void CreateInverseGravity()
	{
		inv_gravity_trans_ = -(camera_gravity_co_.transpose() * camera_gravity_pos_);
		inv_gravity_rot_ = camera_gravity_co_.transpose();
	}

	// joint angles
	void PoseToJointAngles()
	{
		// pelvis
		pelvis_co_ = pelvis_base_co_;
		Eigen::Vector3f pelvis_vec = model_pose_points_[23] - model_pose_points_[24];
		Eigen::Vector3f pelvis_center = 0.5 * (model_pose_points_[23] + model_pose_points_[24]);
		Eigen::Vector3f shoulder_center = 0.5 * (model_pose_points_[11] + model_pose_points_[12]);
		Eigen::Vector3f shoulder_to_pelvis_vec = pelvis_center - shoulder_center;
		float norm_pelvis_vec = pelvis_vec.norm();
		float norm_shoulder_to_pelvis_vec = shoulder_to_pelvis_vec.norm();
		if (norm_pelvis_vec > 1e-5) {
			Eigen::Vector3f pelvis_x = pelvis_vec / norm_pelvis_vec;
			Eigen::Vector3f minus_root_z;
			if (norm_shoulder_to_pelvis_vec > 1e-5) {
				minus_root_z =
					shoulder_to_pelvis_vec / norm_shoulder_to_pelvis_vec;
			}
			else {
				minus_root_z = -Eigen::Vector3f::UnitZ();
			}
			// Eigen::Vector3f pelvis_z = minus_root_z.cross(pelvis_x);
			Eigen::Vector3f pelvis_z = pelvis_x.cross(minus_root_z);
			float norm_pelvis_z = pelvis_z.norm();
			if (norm_pelvis_z > 1e-5) {
				pelvis_z /= norm_pelvis_z;
				// Eigen::Vector3f pelvis_y = pelvis_x.cross(pelvis_z);
				Eigen::Vector3f pelvis_y = pelvis_z.cross(pelvis_x);
				if (mode_ == CoordinatesSet::FBX) {
					MakeMatrixFromVec(pelvis_co_, pelvis_x, pelvis_y, pelvis_z);
				} else if (mode_ == CoordinatesSet::VRM) {
					MakeMatrixFromVec(pelvis_co_, pelvis_x, pelvis_z, -pelvis_y);
				}
			}
		}

		// shoulder
		shoulder_co_ = shoulder_base_co_;
		shoulder_ub_co_ = shoulder_base_co_;
		Eigen::Vector3f shoulder_vec = model_pose_points_[11] - model_pose_points_[12];
		float norm_shoulder_vec = shoulder_vec.norm();
		if (norm_shoulder_vec > 1e-5) {
			Eigen::Vector3f shoulder_x = shoulder_vec / norm_shoulder_vec;

			// Whole body mode, using estimated pelvis coordinate
			if (mode_ == CoordinatesSet::FBX) {
				Eigen::Vector3f pelvis_y = pelvis_co_.col(1);
				// Eigen::Vector3f shoulder_z = pelvis_y.cross(shoulder_x);
				Eigen::Vector3f shoulder_z = shoulder_x.cross(pelvis_y);
				float norm_shoulder_z = shoulder_z.norm();
				if (norm_shoulder_z > 1e-5) {
					shoulder_z /= norm_shoulder_z;
					// Eigen::Vector3f shoulder_y = shoulder_x.cross(shoulder_z);
					Eigen::Vector3f shoulder_y = shoulder_z.cross(shoulder_x);
					MakeMatrixFromVec(shoulder_co_, shoulder_x, shoulder_y, shoulder_z);
				}
			} else if (mode_ == CoordinatesSet::VRM) {
				Eigen::Vector3f pelvis_z = pelvis_co_.col(2);
				Eigen::Vector3f shoulder_y = pelvis_z.cross(shoulder_x);
				float norm_shoulder_y = shoulder_y.norm();
				if (norm_shoulder_y > 1e-5) {
					shoulder_y /= norm_shoulder_y;
					Eigen::Vector3f shoulder_z = shoulder_x.cross(shoulder_y);
					MakeMatrixFromVec(shoulder_co_, shoulder_x, shoulder_y, shoulder_z);
				}
			}

			// Upper body mode, using fixed pelvis coordinate
			if (mode_ == CoordinatesSet::FBX) {
				Eigen::Vector3f pelvis_ub_y = pelvis_base_co_.col(1);
				Eigen::Vector3f shoulder_ub_z = shoulder_x.cross(pelvis_ub_y);
				float norm_shoulder_ub_z = shoulder_ub_z.norm();
				if (norm_shoulder_ub_z > 1e-5) {
					shoulder_ub_z /= norm_shoulder_ub_z;
					Eigen::Vector3f shoulder_ub_y = shoulder_ub_z.cross(shoulder_x);
					MakeMatrixFromVec(shoulder_ub_co_, shoulder_x, shoulder_ub_y, shoulder_ub_z);
				}
			} else if (mode_ == CoordinatesSet::VRM) {
				Eigen::Vector3f pelvis_ub_z = pelvis_base_co_.col(2);
				Eigen::Vector3f shoulder_ub_y = pelvis_ub_z.cross(shoulder_x);
				float norm_shoulder_ub_y = shoulder_ub_y.norm();
				if (norm_shoulder_ub_y > 1e-5) {
					shoulder_ub_y /= norm_shoulder_ub_y;
					Eigen::Vector3f shoulder_ub_z = shoulder_x.cross(shoulder_ub_y);
					MakeMatrixFromVec(shoulder_ub_co_, shoulder_x, shoulder_ub_y, shoulder_ub_z);
				}
			}
		}

		// upper arm
		upper_arm_l_co_ = upper_arm_l_base_co_;
		upper_arm_r_co_ = upper_arm_r_base_co_;
		Eigen::Vector3f upper_arm_l_vec = model_pose_points_[11] - model_pose_points_[13];
		Eigen::Vector3f upper_arm_r_vec = model_pose_points_[12] - model_pose_points_[14];
		MakeUpperArmCoords(upper_arm_l_co_, upper_arm_l_vec, shoulder_co_);
		MakeUpperArmCoords(upper_arm_r_co_, upper_arm_r_vec, shoulder_co_, true);
		// offset
		if (mode_ == CoordinatesSet::FBX) {
			float upper_arm_roll_offset_angle =
				upper_arm_roll_offset_ * shoulder_co_.col(2).dot(Eigen::Vector3f::UnitY());
			upper_arm_l_co_ = upper_arm_l_co_ *
				Eigen::AngleAxisf(Radians(-upper_arm_roll_offset_angle), Eigen::Vector3f::UnitZ());
			upper_arm_r_co_ = upper_arm_r_co_ *
				Eigen::AngleAxisf(Radians(upper_arm_roll_offset_angle), Eigen::Vector3f::UnitZ());
		} else if (mode_ == CoordinatesSet::VRM) {
			float upper_arm_roll_offset_angle =
				upper_arm_roll_offset_ * shoulder_co_.col(1).dot(Eigen::Vector3f::UnitY());
			upper_arm_l_co_ = upper_arm_l_co_ *
				Eigen::AngleAxisf(Radians(-upper_arm_roll_offset_angle), Eigen::Vector3f::UnitY());
			upper_arm_r_co_ = upper_arm_r_co_ *
				Eigen::AngleAxisf(Radians(upper_arm_roll_offset_angle), Eigen::Vector3f::UnitY());
		}

		// elbow
		elbow_l_co_ = elbow_l_base_co_;
		elbow_r_co_ = elbow_r_base_co_;
		Eigen::Vector3f elbow_l_vec = model_pose_points_[13] - model_pose_points_[15];
		Eigen::Vector3f elbow_r_vec = model_pose_points_[14] - model_pose_points_[16];
		MakeElbowCoords(elbow_l_co_, elbow_l_vec, upper_arm_l_co_);
		MakeElbowCoords(elbow_r_co_, elbow_r_vec, upper_arm_r_co_, true);
		// offset
		if (mode_ == CoordinatesSet::FBX) {
			Eigen::AngleAxisf elbow_offset_co(Radians(elbow_offset_), Eigen::Vector3f::UnitX());
			elbow_l_co_ = elbow_l_co_ * elbow_offset_co;
			elbow_r_co_ = elbow_r_co_ * elbow_offset_co;
		} else if (mode_ == CoordinatesSet::VRM) {
			elbow_l_co_ = elbow_l_co_ * Eigen::AngleAxisf(Radians(-elbow_offset_), Eigen::Vector3f::UnitZ());
			elbow_l_co_ = elbow_l_co_ * Eigen::AngleAxisf(Radians(elbow_offset_), Eigen::Vector3f::UnitZ());
		}

		// hand(pose)
		MakePHandCoords(
			phand_l_co_,
			model_pose_points_[15], model_pose_points_[17], model_pose_points_[19],
			elbow_l_co_);
		MakePHandCoords(
			phand_r_co_,
			model_pose_points_[16], model_pose_points_[18], model_pose_points_[20],
			elbow_r_co_, true);

		// thigh
		thigh_l_co_ = thigh_l_base_co_;
		thigh_r_co_ = thigh_r_base_co_;
		Eigen::Vector3f thigh_l_vec = model_pose_points_[23] - model_pose_points_[25];
		Eigen::Vector3f thigh_r_vec = model_pose_points_[24] - model_pose_points_[26];
		MakeThighCoords(thigh_l_co_, thigh_l_vec, pelvis_co_);
		MakeThighCoords(thigh_r_co_, thigh_r_vec, pelvis_co_);
		// offset
		if (mode_ == CoordinatesSet::FBX) {
			float thigh_roll_offset_angle =
				thigh_roll_offset_ * pelvis_co_.col(2).dot(Eigen::Vector3f::UnitY());
			thigh_l_co_ = thigh_l_co_ *
				Eigen::AngleAxisf(Radians(thigh_roll_offset_angle), Eigen::Vector3f::UnitZ());
			thigh_r_co_ = thigh_r_co_ *
				Eigen::AngleAxisf(Radians(-thigh_roll_offset_angle), Eigen::Vector3f::UnitZ());
		} else if (mode_ == CoordinatesSet::VRM) {
			float thigh_roll_offset_angle =
				thigh_roll_offset_ * pelvis_co_.col(1).dot(Eigen::Vector3f::UnitY());
			thigh_l_co_ = thigh_l_co_ *
				Eigen::AngleAxisf(Radians(-thigh_roll_offset_angle), Eigen::Vector3f::UnitY());
			thigh_r_co_ = thigh_r_co_ *
				Eigen::AngleAxisf(Radians(thigh_roll_offset_angle), Eigen::Vector3f::UnitY());
		}

		// knee
		knee_l_co_ = knee_l_base_co_;
		knee_r_co_ = knee_r_base_co_;
		Eigen::Vector3f knee_l_vec = model_pose_points_[25] - model_pose_points_[27];
		Eigen::Vector3f knee_r_vec = model_pose_points_[26] - model_pose_points_[28];
		MakeKneeCoords(knee_l_co_, knee_l_vec, thigh_l_co_);
		MakeKneeCoords(knee_r_co_, knee_r_vec, thigh_r_co_);
		// offset
		Eigen::AngleAxisf knee_offset_co(Radians(knee_offset_), Eigen::Vector3f::UnitX());
		knee_l_co_ = knee_l_co_ * knee_offset_co;
		knee_r_co_ = knee_r_co_ * knee_offset_co;

		// ankle
		MakeAnkleCoords(
			ankle_l_co_,
			model_pose_points_[27], model_pose_points_[29], model_pose_points_[31],
			knee_l_co_);
		MakeAnkleCoords(
			ankle_r_co_,
			model_pose_points_[28], model_pose_points_[30], model_pose_points_[32],
			knee_r_co_);

		// convert to local coords
		Eigen::Matrix3f local_shoulder_co = pelvis_co_.transpose() * shoulder_co_;
		Eigen::Matrix3f local_shoulder_ub_co = pelvis_base_co_.transpose() * shoulder_ub_co_;

		Eigen::Matrix3f local_upper_arm_l_co = shoulder_co_.transpose() * upper_arm_l_co_;
		Eigen::Matrix3f local_elbow_l_co = upper_arm_l_co_.transpose() * elbow_l_co_;
		Eigen::Matrix3f local_phand_l_co = elbow_l_co_.transpose() * phand_l_co_;

		Eigen::Matrix3f local_upper_arm_r_co = shoulder_co_.transpose() * upper_arm_r_co_;
		Eigen::Matrix3f local_elbow_r_co = upper_arm_r_co_.transpose() * elbow_r_co_;
		Eigen::Matrix3f local_phand_r_co = elbow_r_co_.transpose() * phand_r_co_;

		Eigen::Matrix3f local_thigh_l_co = pelvis_co_.transpose() * thigh_l_co_;
		Eigen::Matrix3f local_knee_l_co = thigh_l_co_.transpose() * knee_l_co_;
		Eigen::Matrix3f local_ankle_l_co = knee_l_co_.transpose() * ankle_l_co_;

		Eigen::Matrix3f local_thigh_r_co = pelvis_co_.transpose() * thigh_r_co_;
		Eigen::Matrix3f local_knee_r_co = thigh_r_co_.transpose() * knee_r_co_;
		Eigen::Matrix3f local_ankle_r_co = knee_r_co_.transpose() * ankle_r_co_;

		// diff to local base coords
		shoulder_rot_ = local_shoulder_base_co_.transpose() * local_shoulder_co;
		shoulder_ub_rot_ = local_shoulder_base_co_.transpose() * local_shoulder_ub_co;

		upper_arm_l_rot_ = local_upper_arm_l_base_co_.transpose() * local_upper_arm_l_co;
		elbow_l_rot_ = local_elbow_l_base_co_.transpose() * local_elbow_l_co;
		phand_l_rot_ = local_phand_l_base_co_.transpose() * local_phand_l_co;

		upper_arm_r_rot_ = local_upper_arm_r_base_co_.transpose() * local_upper_arm_r_co;
		elbow_r_rot_ = local_elbow_r_base_co_.transpose() * local_elbow_r_co;
		phand_r_rot_ = local_phand_r_base_co_.transpose() * local_phand_r_co;

		pelvis_rot_ = pelvis_base_co_.transpose() * pelvis_co_;

		thigh_l_rot_ = local_thigh_l_base_co_.transpose() * local_thigh_l_co;
		knee_l_rot_ = local_knee_l_base_co_.transpose() * local_knee_l_co;
		ankle_l_rot_ = local_ankle_l_base_co_.transpose() * local_ankle_l_co;

		thigh_r_rot_ = local_thigh_r_base_co_.transpose() * local_thigh_r_co;
		knee_r_rot_ = local_knee_r_base_co_.transpose() * local_knee_r_co;
		ankle_r_rot_ = local_ankle_r_base_co_.transpose() * local_ankle_r_co;
	}

	/// @brief make limb-root (upperarm, thigh) coords
	/// @param[out] limb_co target coords
	/// @param[in] limb_vec vector from limb to root
	/// @param[in] root_co root coords as parent
	void MakeLimbRootCoords(
		Eigen::Matrix3f& limb_co,
		const Eigen::Vector3f& limb_vec,
		const Eigen::Matrix3f& root_co,
		bool reverse_x = false)
	{
		float norm_limb_vec = limb_vec.norm();
		if (norm_limb_vec > 1e-5) {
			Eigen::Vector3f limb_y = limb_vec / norm_limb_vec;
			Eigen::Vector3f root_x = -root_co.col(0);

			// NOTE: thigh coords are opposite side in y axis
			// Eigen::Vector3f limb_z = limb_y.cross(limb_x);
			Eigen::Vector3f limb_z = reverse_x ?
				limb_y.cross(root_x) : root_x.cross(limb_y);
			float norm_limb_z = limb_z.norm();
			if (norm_limb_z > 1e-5) {
				limb_z /= norm_limb_z;
				// Eigen::Vector3f limb_x = limb_z.cross(limb_y);
				Eigen::Vector3f limb_x = limb_y.cross(limb_z);
				// NOTE: for thigh
				if (reverse_x && limb_x.dot(root_x) > 0) {
					limb_x = -limb_x;
					limb_z = -limb_z;
				}
				// NOTE: if upperarm is raised above shoulder, flip x and z
				if (!reverse_x && limb_y.dot(root_co.col(1)) > 0.0) {
					limb_x = -limb_x;
					limb_z = -limb_z;
				}
				MakeMatrixFromVec(limb_co, limb_x, limb_y, limb_z);
			}
			else {
				Eigen::Vector3f root_y = root_co.col(1);
				limb_z = root_y.cross(limb_y);
				MakeMatrixFromVec(limb_co, root_y, limb_y, limb_z);
			}
		}
	}

	/// @brief make middle joint (elbow, knee) coords
	/// @param[out] limb_co target coords
	/// @param[in] limb_vec vector from limb end to parent end
	/// @param[in] parent_x parent x vector
	/// @param[in] parent_y parent y vector
	void MakeMiddleJointCoords(
		Eigen::Matrix3f& limb_co,
		const Eigen::Vector3f& limb_vec,
		const Eigen::Vector3f& parent_x,
		const Eigen::Vector3f& parent_y,
		bool reverse_x = false)
	{
		float norm_limb_vec = limb_vec.norm();
		if (norm_limb_vec > 1e-5) {
			Eigen::Vector3f limb_y = limb_vec / norm_limb_vec;

			// NOTE: knee coords are opposite side in y axis
			// Eigen::Vector3f limb_x = parent_y.cross(limb_y);
			Eigen::Vector3f limb_x = reverse_x ?
				parent_y.cross(limb_y) : limb_y.cross(parent_y);
			float norm_limb_x = limb_x.norm();
			if (norm_limb_x > 1e-5) {
				limb_x /= norm_limb_x;
				if (limb_x.dot(parent_x) < 0.0) {
					limb_x = -limb_x;
				}
				// Eigen::Vector3f limb_z = limb_y.cross(limb_x);
				Eigen::Vector3f limb_z = limb_x.cross(limb_y);
				MakeMatrixFromVec(limb_co, limb_x, limb_y, limb_z);
			}
		}
	}

	/// @brief make eef (hand(pose), ankle) coords
	/// @param[out] eef_co target coords
	/// @param[in] pt0 point[0]
	/// @param[in] pt1 point[1]
	/// @param[in] pt2 point[2]
	/// @param[in] parent_co parent coords
	void MakeEEFCoords(
		Eigen::Matrix3f& eef_co,
		const Eigen::Vector3f& pt0,
		const Eigen::Vector3f& pt1,
		const Eigen::Vector3f& pt2,
		const Eigen::Matrix3f& parent_co)
	{
		Eigen::Vector3f dir_vec = pt2 - pt1;
		Eigen::Vector3f root_vec = pt0 - pt1;

		float dir_length = dir_vec.norm();
		float root_length = root_vec.norm();

		if (dir_length > 1e-5 && root_length > 1e-5) {
			Eigen::Vector3f eef_z = dir_vec / dir_length;
			Eigen::Vector3f eef_x = root_vec.cross(eef_z);
			eef_x /= eef_x.norm();
			Eigen::Vector3f eef_y = eef_z.cross(eef_x);
			MakeMatrixFromVec(eef_co, eef_x, eef_y, eef_z);
		}
		else {
			eef_co = parent_co;
		}
	}

	/// @brief make upper arm coords
	/// @param[out] upper_arm_co target coords
	/// @param[in] upper_arm_vec vector from elbow to shoulder
	/// @param[in] shoulder_co shoulder coords as parent
	void MakeUpperArmCoords(
		Eigen::Matrix3f& upper_arm_co,
		const Eigen::Vector3f& upper_arm_vec,
		const Eigen::Matrix3f& shoulder_co,
		bool right_arm = false)
	{
		MakeLimbRootCoords(upper_arm_co, upper_arm_vec, shoulder_co);
		if (mode_ == CoordinatesSet::VRM) {
			Eigen::Matrix3f tmp_co = upper_arm_co;
			if (right_arm) {
				upper_arm_co.col(0) = tmp_co.col(1);
				upper_arm_co.col(1) = tmp_co.col(2);
				upper_arm_co.col(2) = tmp_co.col(0);
			} else {
				upper_arm_co.col(0) = -tmp_co.col(1);
				upper_arm_co.col(1) = tmp_co.col(2);
				upper_arm_co.col(2) = -tmp_co.col(0);
			}
		}
	}

	/// @brief make elbow coords
	/// @param[out] elbow_co target coords
	/// @param[in] elbow_vec vector from wrist to elbow
	/// @param[in] upper_arm_co upper arm coords as parent
	void MakeElbowCoords(
		Eigen::Matrix3f& elbow_co,
		const Eigen::Vector3f& elbow_vec,
		const Eigen::Matrix3f& upper_arm_co,
		bool right_arm = false)
	{
		if (mode_ == CoordinatesSet::FBX) {
			MakeMiddleJointCoords(
				elbow_co, elbow_vec, upper_arm_co.col(0), upper_arm_co.col(1));
		} else if (mode_ == CoordinatesSet::VRM) {
			if (right_arm) {
				MakeMiddleJointCoords(
					elbow_co, elbow_vec, upper_arm_co.col(2), upper_arm_co.col(0));
			} else {
				MakeMiddleJointCoords(
					elbow_co, elbow_vec, -upper_arm_co.col(2), -upper_arm_co.col(0));
			}
			Eigen::Matrix3f tmp_co = elbow_co;
			if (right_arm) {
				elbow_co.col(0) = tmp_co.col(1);
				elbow_co.col(1) = tmp_co.col(2);
				elbow_co.col(2) = tmp_co.col(0);
			} else {
				elbow_co.col(0) = -tmp_co.col(1);
				elbow_co.col(1) = tmp_co.col(2);
				elbow_co.col(2) = -tmp_co.col(0);
			}
		}
	}

	/// @brief make hand(pose) coords
	/// @param[out] phand_co target coords
	/// @param[in] pt0 point[0]
	/// @param[in] pt1 point[1]
	/// @param[in] pt2 point[2]
	/// @param[in] parent_co parent coords
	void MakePHandCoords(
		Eigen::Matrix3f& phand_co,
		const Eigen::Vector3f& pt0,
		const Eigen::Vector3f& pt1,
		const Eigen::Vector3f& pt2,
		const Eigen::Matrix3f& parent_co,
		bool right_arm = false)
	{
		MakeEEFCoords(phand_co, pt0, pt1, pt2, parent_co);
		if (mode_ == CoordinatesSet::VRM) {
			Eigen::Matrix3f tmp_co = phand_co;
			if (right_arm) {
				phand_co.col(0) = tmp_co.col(1);
				phand_co.col(1) = tmp_co.col(2);
				phand_co.col(2) = tmp_co.col(0);
			} else {
				phand_co.col(0) = -tmp_co.col(1);
				phand_co.col(1) = tmp_co.col(2);
				phand_co.col(2) = -tmp_co.col(0);
			}
		}
	}

	/// @brief make thigh coords
	/// @param[out] thigh_co target coords
	/// @param[in] thigh_vec vector from thigh to hip
	/// @param[in] pelvis_co pelvis coords as parent
	void MakeThighCoords(
		Eigen::Matrix3f& thigh_co,
		const Eigen::Vector3f& thigh_vec,
		const Eigen::Matrix3f& pelvis_co)
	{
		MakeLimbRootCoords(thigh_co, thigh_vec, pelvis_co, true);
		if (mode_ == CoordinatesSet::VRM) {
			Eigen::Matrix3f tmp_co = thigh_co;
			thigh_co.col(0) = tmp_co.col(0);
			thigh_co.col(1) = -tmp_co.col(2);
			thigh_co.col(2) = tmp_co.col(1);
		}
	}

	/// @brief make knee coords
	/// @param[out] knee_co target coords
	/// @param[in] knee_vec vector from ankle to knee
	/// @param[in] thigh_co thigh coords as parent
	void MakeKneeCoords(
		Eigen::Matrix3f& knee_co,
		const Eigen::Vector3f& knee_vec,
		const Eigen::Matrix3f& thigh_co)
	{
		if (mode_ == CoordinatesSet::FBX) {
			MakeMiddleJointCoords(
				knee_co, knee_vec, thigh_co.col(0), thigh_co.col(1));
		} else if (mode_ == CoordinatesSet::VRM) {
			MakeMiddleJointCoords(
				knee_co, knee_vec, thigh_co.col(0), thigh_co.col(2));
			Eigen::Matrix3f tmp_co = knee_co;
			knee_co.col(0) = tmp_co.col(0);
			knee_co.col(1) = -tmp_co.col(2);
			knee_co.col(2) = tmp_co.col(1);
		}
	}

	/// @brief make ankle coords
	/// @param[out] ankle_co target coords
	/// @param[in] pt0 point[0]
	/// @param[in] pt1 point[1]
	/// @param[in] pt2 point[2]
	/// @param[in] parent_co parent coords
	void MakeAnkleCoords(
		Eigen::Matrix3f& ankle_co,
		const Eigen::Vector3f& pt0,
		const Eigen::Vector3f& pt1,
		const Eigen::Vector3f& pt2,
		const Eigen::Matrix3f& parent_co)
	{
		//MakeEEFCoords(ankle_co, pt0, pt1, pt2, parent_co);
		Eigen::Vector3f dir_vec;
		Eigen::Vector3f root_vec;
		if (mode_ == CoordinatesSet::FBX) {
			dir_vec = pt0 - pt2;
			root_vec = pt1 - pt0;
		} else if (mode_ == CoordinatesSet::VRM) {
			dir_vec = pt2 - pt1;
			root_vec = pt0 - pt1;
		}

		float dir_length = dir_vec.norm();
		float root_length = root_vec.norm();

		if (dir_length > 1e-5 && root_length > 1e-5) {
			Eigen::Vector3f eef_y = dir_vec / dir_length;
			Eigen::Vector3f eef_x = eef_y.cross(root_vec);
			eef_x /= eef_x.norm();
			Eigen::Vector3f eef_z = eef_x.cross(eef_y);
			MakeMatrixFromVec(ankle_co, eef_x, eef_y, eef_z);
		}
		else {
			// TODO: if asymmetry?
			ankle_co = parent_co * local_ankle_l_base_co_;
		}
	}


	// face params
	FaceParams& face_params()
	{
		return face_params_;
	}

	Eigen::Matrix3f& face_co()
	{
		return face_co_;
	}

	Eigen::Matrix3f& face_rot()
	{
		return face_rot_;
	}

	Eigen::Matrix3f& face_ub_rot()
	{
		return face_ub_rot_;
	}

	void FaceToJointAngles()
	{
		MakeNeckCoords();
		face_params_.FaceToFaceParams(model_face_points_);
	}

	void MakeNeckCoords()
	{
		if (face_params_.MakeNeckCoords(model_face_points_)) {
			// face rotation in the model coordinate
			if (mode_ == CoordinatesSet::FBX) {
				face_co_ = face_params_.face_co();
			} else if (mode_ == CoordinatesSet::VRM) {
				face_co_.col(0) = face_params_.face_co().col(0);
				face_co_.col(1) = face_params_.face_co().col(2);
				face_co_.col(2) = -face_params_.face_co().col(1);
			}
			// adjust pitch offset
			face_co_ = face_co_ * Eigen::AngleAxisf(
				Radians(-20.0 + neck_pitch_offset_), Eigen::Vector3f::UnitX());

			// local rotation for Whole body tracking
			Eigen::Matrix3f local_face_co = shoulder_co_.transpose() * face_co_;
			face_rot_ = local_shoulder_base_co_.transpose() * local_face_co;

			// local rotation for Upper body tracking
			Eigen::Matrix3f local_face_ub_co = shoulder_ub_co_.transpose() * face_co_;
			face_ub_rot_ = local_shoulder_base_co_.transpose() * local_face_ub_co;
		}
	}

	// hand
	HandParams& right_hand_params()
	{
		return right_hand_params_;
	}
	HandParams& left_hand_params()
	{
		return left_hand_params_;
	}

	void RHandToJointAngles()
	{
		right_hand_params_.HandToJointAngles(
			model_right_hand_points_,
			elbow_r_co_,
			local_phand_r_base_co_);
	}

	void LHandToJointAngles()
	{
		left_hand_params_.HandToJointAngles(
			model_left_hand_points_,
			elbow_l_co_,
			local_phand_l_base_co_);
	}

	// offset
	void upper_arm_roll_offset(float val)
	{
		upper_arm_roll_offset_ = val;
	}

	float upper_arm_roll_offset()
	{
		return upper_arm_roll_offset_;
	}

	void elbow_offset(float val)
	{
		elbow_offset_ = val;
	}

	float elbow_offset()
	{
		return elbow_offset_;
	}

	void thigh_roll_offset(float val)
	{
		thigh_roll_offset_ = val;
	}

	float thigh_roll_offset()
	{
		return thigh_roll_offset_;
	}

	void knee_offset(float val)
	{
		knee_offset_ = val;
	}

	float knee_offset()
	{
		return knee_offset_;
	}

	void neck_pitch_offset(float val)
	{
		neck_pitch_offset_ = val;
	}

	float neck_pitch_offset()
	{
		return neck_pitch_offset_;
	}

	//
	Eigen::Vector3f EstimateHipPosition()
	{
		Eigen::Vector3f result(0.0, 0.0, 0.0);

		if (pose_2d_points_.size() != pose_points_.size()) {
			return result;
		}

		size_t num_points = pose_2d_points_.size();
		Eigen::MatrixXf amat(2 * num_points, 3);
		Eigen::VectorXf bvec(2 * num_points);

		for (size_t i = 0; i < num_points; i++) {
			const std::vector<float>& uv = pose_2d_points_[i];
			const std::vector<float>& pt = pose_points_[i];

			// NOTE: pose_2d_points shall be converted into image geometries
			//float u = uv[0] * frame_height_;
			//float v = uv[1] * frame_width_;
			float u = (uv[0] - 0.5) * frame_height_;
			float v = (uv[1] - 0.5) * frame_width_;

			amat(i * 2, 0) = focal_length_;
			amat(i * 2, 1) = 0.0;
			amat(i * 2, 2) = -u;
			amat(i * 2 + 1, 0) = 0.0;
			amat(i * 2 + 1, 1) = focal_length_;
			amat(i * 2 + 1, 2) = -v;

			bvec(i * 2) = u * pt[2] - pt[0] * focal_length_;
			bvec(i * 2 + 1) = v * pt[2] - pt[1] * focal_length_;
		}

		Eigen::VectorXf hippos =
			amat.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(bvec);

		if (hippos.size() == 3) {
			for (size_t i = 0; i < 3; i++) {
				result(i) = hippos(i);
			}
		}

		return result;
	}

	Eigen::Vector3f EstimateShoulderPosition()
	{
		Eigen::Vector3f result(0.0, 0.0, 0.0);

		if (pose_2d_points_.size() != pose_points_.size()) {
			return result;
		}

		const std::vector<float>& left_shoulder = pose_points_[11];
		const std::vector<float>& right_shoulder = pose_points_[12];
		for (size_t i = 0; i< 3; i++) {
			result(i) = (left_shoulder[i] + right_shoulder[i]) * 0.5;
		}

		return result;
	}

};

}  // namespace mediapipe_receiver

#endif  // MEDIAPIPE_RECEIVER_CPP_MEDIAPIPE_RECEIVER_H_
