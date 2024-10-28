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
  // calculate distance

  Eigen::VectorXd distances(trainSamples.rows());

  for (int i = 0; i < trainSamples.rows(); i++) {
    Eigen::VectorXd row = trainSamples.row(i);
    distances(i) = euclideanDistance(features, row);
  }

  // get the closest K neighbours

  std::map<int, double> kIndexes; 

  for (int i = 0; i < distances.rows(); i++) {
    double distance = distances(i);

    if (kIndexes.size() < K) {
      kIndexes.insert(std::pair<int, double>(i, distance));
    } else {
      int indexToReplace = -1;
      double indexToReplaceDistance = 0.0;

      for (std::pair<int, double> index: kIndexes) {
        if (distance < index.second && (indexToReplace == -1 || (indexToReplaceDistance < index.second))) {
          indexToReplace = index.first;
          indexToReplaceDistance = index.second;
        }
      }

      if (indexToReplace > -1) {
        kIndexes.erase(indexToReplace);
        kIndexes.insert(std::pair<int, double>(i, distance));
      }
    }
  }

  // get the labels of the closest K neighbours

  Eigen::VectorXd kLabels(kIndexes.size());

  int i = 0;

  for (std::pair<int, double> index: kIndexes) {
    kLabels(i) = trainLabels(index.first);
    i++;
  }

  // get the label with highest frequency (majority vote)

  return MLUtils::mostCommonLabel(kLabels);
}

double KNN::euclideanDistance(Eigen::VectorXd& test, Eigen::VectorXd& train) {
  Eigen::ArrayXd subtracted = test - train;
  double sum = subtracted.pow(2.0).sum();
  return std::sqrt(sum);
}
