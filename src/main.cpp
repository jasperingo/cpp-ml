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

  // std::cout << "Number of train samples: " << splitResult.trainSamples.rows() << std::endl;
  // std::cout << "Number of test samples: " << splitResult.testSamples.rows() << std::endl;
  // std::cout << "Number of train labels: " << splitResult.trainLabels.rows() << std::endl;
  // std::cout << "Number of test labels: " << splitResult.testLabels.rows() << std::endl;

  std::cout << "Train samples: " << std::endl << splitResult.trainSamples << std::endl;
  std::cout << "Test samples: " << std::endl << splitResult.testSamples << std::endl;
  std::cout << "Train labels: " << std::endl << splitResult.trainLabels << std::endl;
  std::cout << "Test labels: " << std::endl << splitResult.testLabels << std::endl;

  // KNN knn;
}
