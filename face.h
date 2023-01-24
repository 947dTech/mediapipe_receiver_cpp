#pragma once

#ifndef MEDIAPIPE_RECEIVER_CPP_FACE_H_
#define MEDIAPIPE_RECEIVER_CPP_FACE_H_

#include <vector>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include "functions.h"

namespace mediapipe_receiver {

class FaceParams {
private:
  float r_eye_area_;
  float l_eye_area_;
  float mouth_open_;
  float smile_;

  Eigen::Matrix3f face_co_;

public:
  FaceParams()
  {
    r_eye_area_ = 0.0;
    l_eye_area_ = 0.0;
    mouth_open_ = 0.0;
    smile_ = 0.0;
  }

  float r_eye_area()
  {
    return r_eye_area_;
  }

  float l_eye_area()
  {
    return l_eye_area_;
  }

  float mouth_open()
  {
    return mouth_open_;
  }

  float smile()
  {
    return smile_;
  }

  Eigen::Matrix3f& face_co()
  {
    return face_co_;
  }

  bool MakeNeckCoords(std::vector <Eigen::Vector3f>& face_points)
  {
    bool result = false;
    // use camera coordinate
    // face direction, in camera coords
    Eigen::Vector3f r_eye_pt = 0.5 * (face_points[33] + face_points[133]);
    Eigen::Vector3f l_eye_pt = 0.5 * (face_points[263] + face_points[362]);
    Eigen::Vector3f lip_pt = 0.5 * (face_points[13] + face_points[14]);

    Eigen::Vector3f face_x_vec = l_eye_pt - r_eye_pt;
    Eigen::Vector3f eye_to_mouth_vec = lip_pt - 0.5 * (r_eye_pt + l_eye_pt);
    Eigen::Vector3f face_z_vec = face_x_vec.cross(eye_to_mouth_vec);

    float face_x_norm = face_x_vec.norm();
    float face_z_norm = face_z_vec.norm();
    result = face_x_norm > 1e-5 && face_z_norm > 1e-5;
    if (result) {
      face_x_vec /= face_x_norm;
      face_z_vec /= face_z_norm;
      Eigen::Vector3f face_y_vec = face_z_vec.cross(face_x_vec);
      MakeMatrixFromVec(face_co_, face_x_vec, face_y_vec, face_z_vec);
    }
    return result;
  }

  void FaceToFaceParams(std::vector <Eigen::Vector3f>& face_points)
  {
    // face parameters
    // eyes
    float eyes_distance = (face_points[133] - face_points[362]).norm();
    if (eyes_distance > 1e-5) {
      float r_eye_length = (face_points[33] - face_points[133]).norm() / eyes_distance;
      float r_eye_distance = (face_points[145] - face_points[159]).norm() / eyes_distance;
      r_eye_area_ = r_eye_distance * r_eye_length;

      float l_eye_length = (face_points[263] - face_points[362]).norm() / eyes_distance;
      float l_eye_distance = (face_points[374] - face_points[386]).norm() / eyes_distance;
      l_eye_area_ = l_eye_distance * l_eye_length;
    }

    // mouth
    Eigen::Vector3f mouth_vector = (face_points[0] - face_points[17]);
    if (eyes_distance > 1e-5 && mouth_vector.norm() > 1e-5) {
      float lip_distance = (face_points[13] - face_points[14]).norm();
      Eigen::Vector3f lip_edges_center = 0.5 * (face_points[78] + face_points[308]);
      Eigen::Vector3f lip_center_center = 0.5 * (face_points[13] + face_points[14]);
      Eigen::Vector3f lip_edges_center_vector = lip_edges_center - face_points[17];
      Eigen::Vector3f lip_center_center_vector = lip_center_center - face_points[17];

      mouth_open_ = lip_distance / eyes_distance;
      float t_lip_edges_center = mouth_vector.dot(lip_edges_center_vector);
      float t_lip_center_center = mouth_vector.dot(lip_center_center_vector);
      smile_ = (t_lip_edges_center - t_lip_center_center) / eyes_distance;
    }
  }
};

}  // namespace mediapipe_receiver

#endif  // MEDIAPIPE_RECEIVER_CPP_FACE_H_
