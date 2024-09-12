#include "KNN.hpp"

void KNN::fit(Eigen::VectorXd& labels, Eigen::MatrixXd& samples) {
  trainLabels = labels;
  trainSamples = samples;
}

Eigen::VectorXd KNN::predict(Eigen::MatrixXd& samples) {
  Eigen::VectorXd predictions(samples.rows());

  for (int i = 0; i < samples.rows(); i++) {
    Eigen::VectorXd row = samples.row(i);
    predictions(i) = predict(row);
  }

  return predictions;
}

double KNN::predict(Eigen::VectorXd& features) {
  Eigen::VectorXd distances(trainSamples.rows());

  for (int i = 0; i < trainSamples.rows(); i++) {
    Eigen::VectorXd row = trainSamples.row(i);
    distances(i) = euclideanDistance(features, row);
  }

  std::map<int, double> kIndexes; 

  for (int i = 0; i < distances.rows(); i++) {
    double distance = distances(i);

    if (kIndexes.size() < K) {
      kIndexes.insert(std::pair<int, double>(i, distance));
    } else {
      auto it = kIndexes.begin();

      while (it != kIndexes.end()) {
         if (distance < it->second) {
          kIndexes.erase(it);
          kIndexes.insert(std::pair<int, double>(i, distance));
          break;
        } else {
          it++;
        }
      }
    }
  }

  Eigen::VectorXd kLabels(kIndexes.size());

  // for (int i = 0; i < kIndexes.size(); i++) {
  //   kLabels(i) = trainLabels(kIndexes[i].first);
  // }


  return 0.0;
}

double KNN::euclideanDistance(Eigen::VectorXd& test, Eigen::VectorXd& train) {
  Eigen::ArrayXd subtracted = test - train;
  double sum = subtracted.pow(2.0).sum();
  return std::sqrt(sum);
}
