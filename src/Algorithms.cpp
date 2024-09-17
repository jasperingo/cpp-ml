#include "Algorithms.hpp"

int doKNN(
  std::string datasetPath, 
  unsigned int labelColumn, 
  bool labelColumIsDigit, 
  unsigned int sampleColumns[], 
  bool sampleColumnsAreDigits[], 
  size_t sampleColumnsLength
) {
  CSVETL csvETL(datasetPath);

  if (!csvETL.load(labelColumn, labelColumIsDigit, sampleColumns, sampleColumnsAreDigits, sampleColumnsLength)) {
    std::cout << "Dataset: " << datasetPath << " not loaded " << std::endl;
    return 1;
  }

  std::cout << "Dataset: " << datasetPath << " loaded " << std::endl;

  Eigen::MatrixXd X = csvETL.getSamples();

  Eigen::VectorXd y = csvETL.getLabels();

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

  KNN knn(3);

  knn.fit(splitResult.trainLabels, splitResult.trainSamples);

  Eigen::VectorXd predictions = knn.predict(splitResult.testSamples);

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

  return 0;
}
