#pragma once

#ifndef MEDIAPIPE_RECEIVER_CPP_HAND_H_
#define MEDIAPIPE_RECEIVER_CPP_HAND_H_

#include <vector>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <cmath>
#include "functions.h"

namespace mediapipe_receiver {

class HandParams {
private:
  std::vector<float> thumb_angles_;
  std::vector<float> index_finger_angles_;
  std::vector<float> middle_finger_angles_;
  std::vector<float> ring_finger_angles_;
  std::vector<float> pinky_angles_;

  float thumb_side_angle_;
  float index_finger_side_angle_;
  float middle_finger_side_angle_;
  float ring_finger_side_angle_;
  float pinky_side_angle_;

  Eigen::Matrix3f hand_co_;
  Eigen::Matrix3f hand_rot_;

  CoordinatesSet mode_;
  bool right_hand_;

public:
  HandParams()
  {
    mode_ = CoordinatesSet::FBX;
    right_hand_ = false;

    thumb_angles_.resize(3);
    index_finger_angles_.resize(3);
    middle_finger_angles_.resize(3);
    ring_finger_angles_.resize(3);
    pinky_angles_.resize(3);

    for (size_t i = 0; i < 3; i++) {
      thumb_angles_[i] = 0.0;
      index_finger_angles_[i] = 0.0;
      middle_finger_angles_[i] = 0.0;
      ring_finger_angles_[i] = 0.0;
      pinky_angles_[i] = 0.0;
    }

    thumb_side_angle_ = 0.0;
    index_finger_side_angle_ = 0.0;
    middle_finger_side_angle_ = 0.0;
    ring_finger_side_angle_ = 0.0;
    pinky_side_angle_ = 0.0;
  }

  void SetRightHand()
  {
    right_hand_ = true;
  }

  void SetLeftHand()
  {
    right_hand_ = false;
  }

  void SetMode(CoordinatesSet mode)
  {
    mode_ = mode;
  }

  std::vector<float>& thumb_angles()
  {
    return thumb_angles_;
  }

  std::vector<float>& index_finger_angles()
  {
    return index_finger_angles_;
  }

  std::vector<float>& middle_finger_angles()
  {
    return middle_finger_angles_;
  }

  std::vector<float>& ring_finger_angles()
  {
    return ring_finger_angles_;
  }

  std::vector<float>& pinky_angles()
  {
    return pinky_angles_;
  }

  float thumb_side_angle()
  {
    return thumb_side_angle_;
  }

  float index_finger_side_angle()
  {
    return index_finger_side_angle_;
  }

  float middle_finger_side_angle()
  {
    return middle_finger_side_angle_;
  }

  float ring_finger_side_angle()
  {
    return ring_finger_side_angle_;
  }

  float pinky_side_angle()
  {
    return pinky_side_angle_;
  }

  Eigen::Matrix3f& hand_co() {
    return hand_co_;
  }

  Eigen::Matrix3f& hand_rot() {
    return hand_rot_;
  }

  /// @brief make hand coordinate (common for both hands)
  /// @param[in] hand_points hand key points
  /// @param[out] hand_co hand rotation matrix
  /// @return false if coordinate can not be estimated
  bool MakeHandCoords(
    std::vector <Eigen::Vector3f>& hand_points,
    Eigen::Matrix3f& hand_co)
  {
    bool result = false;
    //Eigen::Vector3f hand_slide_vector = hand_points[5] - hand_points[17];  // z
    Eigen::Vector3f hand_slide_vector = hand_points[17] - hand_points[5];  // z
    Eigen::Vector3f fingers_root_center = (
      hand_points[5] +
      hand_points[9] +
      hand_points[13] +
      hand_points[17]) * 0.25;
    Eigen::Vector3f hand_dir_vector = hand_points[0] - fingers_root_center;  // y
    float norm_slide = hand_slide_vector.norm();
    float norm_dir = hand_dir_vector.norm();
    if (norm_slide > 1e-5 && norm_dir > 1e-5) {
      hand_slide_vector /= norm_slide;
      hand_dir_vector /= norm_dir;

      Eigen::Vector3f hand_normal_vector = hand_slide_vector.cross(hand_dir_vector);  // x
      float norm_normal = hand_normal_vector.norm();
      if (norm_normal > 1e-5) {
        hand_normal_vector /= norm_normal;
        hand_slide_vector = hand_normal_vector.cross(hand_dir_vector);
        if (mode_ == CoordinatesSet::FBX) {
          MakeMatrixFromVec(hand_co, hand_normal_vector, hand_dir_vector, hand_slide_vector);
        } else if (mode_ == CoordinatesSet::VRM) {
          if (right_hand_) {
            MakeMatrixFromVec(hand_co, hand_dir_vector, hand_slide_vector, hand_normal_vector);
          } else {
            MakeMatrixFromVec(hand_co, -hand_dir_vector, hand_slide_vector, -hand_normal_vector);
          }
        }
        result = true;
      }
    }
    return result;
  }

  /// @brief calc angle of pt0 --- pt1 --- pt2 hinge
  /// @param pt0 end point
  /// @param pt1 center point
  /// @param pt2 another end point
  /// @return angle of hinge
  float FingerAngle(
    const Eigen::Vector3f& pt0,
    const Eigen::Vector3f& pt1,
    const Eigen::Vector3f& pt2)
  {
    float angle = 0.0;

    Eigen::Vector3f v10 = pt1 - pt0;
    Eigen::Vector3f v21 = pt2 - pt1;

    float norm_v10 = v10.norm();
    float norm_v21 = v21.norm();

    if (norm_v10 > 1e-5 && norm_v21 > 1e-5) {
      v10 /= norm_v10;
      v21 /= norm_v21;
      angle = acos(v10.dot(v21));
    }

    return angle;
  }

  /// @brief calc angle between basis_vec and pt0 --- pt1
  /// @param basis_vec basis vector, normalized [5] - [17]
  /// @param pt0 end point
  /// @param pt1 another end point
  /// @return side angle of finger
  float FingerSideAngle(
    const Eigen::Vector3f& basis_vec,
    const Eigen::Vector3f& pt0,
    const Eigen::Vector3f& pt1
  )
  {
    float angle = 0.0;

    Eigen::Vector3f v10 = pt1 - pt0;
    float norm_v10 = v10.norm();
    if (norm_v10 > 1e-5) {
      v10 /= norm_v10;
      angle = asin(v10.dot(basis_vec));
    }

    return angle;
  }

  void HandToJointAngles(
    std::vector <Eigen::Vector3f>& hand_points,
    const Eigen::Matrix3f& elbow_co,
    const Eigen::Matrix3f& local_phand_base_co)
  {
    hand_co_ = Eigen::Matrix3f::Identity();
    bool hand_detected = MakeHandCoords(hand_points, hand_co_);

    if (hand_detected) {
      Eigen::Matrix3f local_hand_co = elbow_co.transpose() * hand_co_;
      hand_rot_ = local_phand_base_co.transpose() * local_hand_co;
      // finger angles, root to tip
      size_t index_finger_indices[] = { 0, 5, 6, 7, 8 };
      size_t middle_finger_indices[] = { 0, 9, 10, 11, 12 };
      size_t ring_finger_indices[] = { 0, 13, 14, 15, 16 };
      size_t pinky_indices[] = { 0, 17, 18, 19, 20 };
      for (size_t i = 0; i < 3; i++) {
        // index
        index_finger_angles_[i] = FingerAngle(
          hand_points[index_finger_indices[i]],
          hand_points[index_finger_indices[i + 1]],
          hand_points[index_finger_indices[i + 2]]);
        // middle
        middle_finger_angles_[i] = FingerAngle(
          hand_points[middle_finger_indices[i]],
          hand_points[middle_finger_indices[i + 1]],
          hand_points[middle_finger_indices[i + 2]]);
        // ring
        ring_finger_angles_[i] = FingerAngle(
          hand_points[ring_finger_indices[i]],
          hand_points[ring_finger_indices[i + 1]],
          hand_points[ring_finger_indices[i + 2]]);
        pinky_angles_[i] = FingerAngle(
          hand_points[pinky_indices[i]],
          hand_points[pinky_indices[i + 1]],
          hand_points[pinky_indices[i + 2]]);
      }

      // side angles
      // NOTE: all side angles are negative in Left hand
      Eigen::Vector3f basis_vec = hand_co_.col(2);
      Eigen::Vector3f appr_vec = hand_co_.col(1);
      if (mode_ == CoordinatesSet::FBX) {
        // nop
      }  else if (mode_ == CoordinatesSet::VRM) {
          basis_vec = hand_co_.col(1);
          if (right_hand_) {
            appr_vec = hand_co_.col(0);
          } else {
            appr_vec = -hand_co_.col(0);
          }
      }

      index_finger_side_angle_ = (index_finger_angles_[0] < 0.5) ?
        FingerSideAngle(basis_vec, hand_points[5], hand_points[6]) : 0.0;
      middle_finger_side_angle_ = (middle_finger_angles_[0] < 0.5) ?
        FingerSideAngle(basis_vec, hand_points[9], hand_points[10]) : 0.0;
      ring_finger_side_angle_ = (ring_finger_angles_[0] < 0.5) ?
        FingerSideAngle(basis_vec, hand_points[13], hand_points[14]) : 0.0;
      pinky_side_angle_ = (pinky_angles_[0] < 0.5) ?
        FingerSideAngle(basis_vec, hand_points[17], hand_points[18]) : 0.0;

      // Thumb angle may be negative value
      float t0 = FingerAngle(hand_points[0], hand_points[1], hand_points[2]);
      float t1 = FingerAngle(hand_points[1], hand_points[2], hand_points[3]);
      float t2 = FingerAngle(hand_points[2], hand_points[3], hand_points[4]);
      Eigen::Vector3f t1_dir = hand_points[2] - (hand_points[1] + hand_points[3]) * 0.5;
      Eigen::Vector3f t2_dir = hand_points[3] - (hand_points[2] + hand_points[4]) * 0.5;

      float tan_t1 = atan2(t1_dir.dot(basis_vec), t1_dir.dot(appr_vec));
      float tan_t2 = atan2(t2_dir.dot(basis_vec), t2_dir.dot(appr_vec));
      thumb_angles_[1] = (tan_t1 > 0.0) ? t1 : -t1;
      thumb_angles_[2] = (tan_t2 > 0.0) ? t2 : -t2;
      Eigen::Vector3f thumb_dir_vec = hand_points[1] - hand_points[2];
      float norm_tdir = thumb_dir_vec.norm();
      if (norm_tdir > 1e-5) {
        thumb_dir_vec /= norm_tdir;
        thumb_angles_[0] = asin(thumb_dir_vec.dot(basis_vec));
        thumb_side_angle_ = acos(thumb_dir_vec.dot(appr_vec));
      }
    }
  }
};

}  // namespace mediapipe_receiver

#endif  // MEDIAPIPE_RECEIVER_CPP_HAND_H_
