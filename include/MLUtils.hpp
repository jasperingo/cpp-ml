#ifndef CPP_ML_ML_UTILS_H_
#define CPP_ML_ML_UTILS_H_

#include <map>
#include <tuple>
#include <iostream>
#include <Eigen/Dense>

namespace MLUtils {
  std::tuple<double, unsigned int, unsigned int> calculateAccuracy(Eigen::VectorXd predictions, Eigen::VectorXd labels) {
    unsigned int correctCount = 0;
    unsigned int incorrectCount = 0;

    for (int i = 0; i < predictions.size(); i++) {
      double prediction = predictions(i);
      double label = labels(i);

      if (prediction == label) {
        correctCount++;
      } else {
        incorrectCount++;
      }
    }

    double accuracy = ((double) correctCount) / labels.size();

    return std::tuple<double, unsigned int, unsigned int>(accuracy, correctCount, incorrectCount);
  }

  void calculateAndPrintAccuracy(Eigen::VectorXd predictions, Eigen::VectorXd labels) {
    std::tuple<double, unsigned int, unsigned int> result = calculateAccuracy(predictions, labels);
    double accuracy = std::get<0>(result);
    unsigned int correctCount = std::get<1>(result);
    unsigned int incorrectCount = std::get<2>(result);

    std::cout << "Model accuracy: " << accuracy << std::endl;
    std::cout << "Model accuracy %: " << (accuracy * 100) << std::endl;
    std::cout << "Number of correct predictions: " << correctCount << std::endl;
    std::cout << "Number of incorrect predictions: " << incorrectCount << std::endl;
  }

  std::map<double, unsigned int> labelsFrequency(Eigen::VectorXd &labels) {
    std::map<double, unsigned int> frequency;

    for (int i = 0; i < labels.rows(); i++) {
      double label = labels(i);

      std::map<double, unsigned int>::iterator it = frequency.find(label);

      if (it == frequency.end()) {
        frequency.insert(std::pair<double, unsigned int>(label, 1));
      } else {
        it->second = it->second + 1;
      }
    }

    return frequency;
  }

  double mostCommonLabel(Eigen::VectorXd &labels) {
    std::map<double, unsigned int> labelsWithFrequency = labelsFrequency(labels);

    // get the label with highest frequency (majority vote)

    double mostCommonLabel = 0.0;
    unsigned int mostCommonLabelCount = 0;

    for (std::pair<double, unsigned int> label: labelsWithFrequency) {
      if (label.second > mostCommonLabelCount) {
          mostCommonLabel = label.first;
          mostCommonLabelCount = label.second;
      }
    }

    return mostCommonLabel;
  }

}

#endif /* CPP_ML_ML_UTILS_H_ */
