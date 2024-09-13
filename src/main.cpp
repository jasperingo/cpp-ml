#include <iostream>
#include <Eigen/Dense>
#include "KNN.hpp"
#include "CSVETL.hpp"

int main(int argc, char* argv[]) {
   if (argc < 2) {
    std::cout << "Name of CSV file not provided" << std::endl;
    return 1;
  }

  CSVETL csvETL(argv[1]);

  if (!csvETL.load()) {
    std::cout << "Dataset: " << argv[1] << " not loaded " << std::endl;
    return 1;
  }

  std::cout << "Dataset: " << argv[1] << " loaded " << std::endl;

  size_t sampleColumns[] = { 1, 2, 3, 4 };

  Eigen::MatrixXd X = csvETL.extractSamples(sampleColumns, 4);

  Eigen::VectorXd y = csvETL.extractLabels(5);

  // std::cout << "Samples:" << std::endl << std::endl;

  // std::cout << X << std::endl << std::endl;

  // std::cout << "Labels:" << std::endl << std::endl;

  // std::cout << y << std::endl << std::endl;

  std::cout << "Number of samples: " << X.rows() << std::endl;
  std::cout << "Number of features: " << X.cols() << std::endl;

  CSVETL::DataSplitResult splitResult = csvETL.splitData(y, X);

  std::cout << "Number of train samples: " << splitResult.trainSamples.rows() << std::endl;
  std::cout << "Number of test samples: " << splitResult.testSamples.rows() << std::endl;
  std::cout << "Number of train labels: " << splitResult.trainLabels.rows() << std::endl;
  std::cout << "Number of test labels: " << splitResult.testLabels.rows() << std::endl;

  // std::cout << "Train samples: " << std::endl << splitResult.trainSamples << std::endl;
  // std::cout << "Test samples: " << std::endl << splitResult.testSamples << std::endl;
  // std::cout << "Train labels: " << std::endl << splitResult.trainLabels << std::endl;
  // std::cout << "Test labels: " << std::endl << splitResult.testLabels << std::endl;

  // std::cout << "Row 1: " << std::endl << row1 << std::endl;
  // std::cout << "Row to Pow: " << std::endl << row0ToPow << std::endl;
  // std::cout << "Row to Pow summed: " << std::endl << (row0ToPow.sum()) << std::endl;

  KNN knn(3);

  knn.fit(splitResult.trainLabels, splitResult.trainSamples);

  Eigen::VectorXd predictions = knn.predict(splitResult.testSamples);

  std::cout << "Predictions: " << std::endl << predictions << std::endl;
  std::cout << "Number of predictions: " << predictions.rows() << std::endl;

  int correctCount = 0;
  int incorrectCount = 0;

  for (int i = 0; i < predictions.rows(); i++) {
    double prediction = predictions(i);
    double testLabel = splitResult.testLabels(i);

    if (prediction == testLabel) {
      correctCount++;
    } else {
      incorrectCount++;
    }
  }

  double accuracy = ((double) correctCount) / splitResult.testLabels.rows();

  std::cout << "Model accuracy: " << accuracy << std::endl;
  std::cout << "Model accuracy %: " << (accuracy * 100) << std::endl;
  std::cout << "Number of correct predictions: " << correctCount << std::endl;
  std::cout << "Number of incorrect predictions: " << incorrectCount << std::endl;
}
