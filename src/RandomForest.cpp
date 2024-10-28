#include "RandomForest.hpp"

std::tuple<Eigen::MatrixXd, Eigen::VectorXd, std::vector<unsigned int>> RandomForest::bootstrap(Eigen::MatrixXd &features, Eigen::VectorXd &labels) {
  unsigned int numberOfDataSamples = (unsigned int) features.rows();
  unsigned int numberOfDataFeatures = (unsigned int) features.cols();
  
  unsigned int numberOfSamplesToSelect = numberOfDataSamples <= 20 ? numberOfDataSamples : numberOfDataSamples - 10;
  unsigned int numberOfFeaturesToSelect = numberOfDataFeatures <= 2 ? numberOfDataFeatures : numberOfDataFeatures - 1;

  unsigned int numberOfFoundSamples = 0;
  unsigned int numberOfFoundFeatures = 0;

  std::vector<unsigned int> sampleIndexes(numberOfSamplesToSelect);
  std::vector<unsigned int> featureIndexes(numberOfFeaturesToSelect);
  
  std::cout << "Bootstrapping features" << std::endl;

  while (numberOfFoundFeatures < numberOfFeaturesToSelect) {
    std::srand((unsigned int) std::time(0)); // use current time as seed for random generator
    int randomIndex = std::rand() % numberOfDataFeatures;  // Modulo to restrict the number of random values to be at most vector.size()-1
    std::vector<unsigned int>::iterator it = std::find(featureIndexes.begin(), featureIndexes.end(), randomIndex);

    if (it == featureIndexes.end()) {
      featureIndexes.at(numberOfFoundFeatures) = randomIndex;
      numberOfFoundFeatures++;
    }
  }

  // std::cout << "Bootstrapping features count: " << featureIndexes.size() << 
  //   " & number of features: " << numberOfDataFeatures << 
  //   " & number of features to select: " << numberOfFeaturesToSelect << std::endl;

  std::cout << "Bootstrapping samples" << std::endl;

  unsigned int numberOfSamplesAfterRandomIndex = numberOfSamplesToSelect > 50 
    ? 50 
    : numberOfSamplesToSelect < 20 ? numberOfSamplesToSelect : numberOfSamplesToSelect - 10;

  while (numberOfFoundSamples < numberOfSamplesToSelect) {
    std::srand((unsigned int) std::time(0)); // use current time as seed for random generator
    unsigned int randomIndex = std::rand() % numberOfDataSamples;  // Modulo to restrict the number of random values to be at most vector.size()-1
    std::vector<unsigned int>::iterator it = std::find(sampleIndexes.begin(), sampleIndexes.end(), randomIndex);

    if (it == sampleIndexes.end()) {
      sampleIndexes.at(numberOfFoundSamples) = randomIndex;
      numberOfFoundSamples++;
    }

    for (unsigned int i = 0; i < numberOfSamplesAfterRandomIndex; i++) {
      if (numberOfFoundSamples >= numberOfSamplesToSelect) {
        break;
      }

      randomIndex++;

      if (randomIndex >= numberOfDataSamples) {
        randomIndex = 0;
      }

      it = std::find(sampleIndexes.begin(), sampleIndexes.end(), randomIndex);

      if (it == sampleIndexes.end()) {
        sampleIndexes.at(numberOfFoundSamples) = randomIndex;
        numberOfFoundSamples++;
      }
    }
  }

  Eigen::VectorXd labelsSubset(sampleIndexes.size());
  Eigen::MatrixXd featuresSubset(sampleIndexes.size(), featureIndexes.size());

  for (unsigned int i = 0; i < sampleIndexes.size(); i++) {
    labelsSubset(i) = labels(sampleIndexes.at(i));

    for (unsigned int j = 0; j < featureIndexes.size(); j++) {
      featuresSubset(i, j) = features(sampleIndexes.at(i), featureIndexes.at(j));
    }
  }

  return std::tuple<Eigen::MatrixXd, Eigen::VectorXd, std::vector<unsigned int>>(featuresSubset, labelsSubset, featureIndexes);
}

void RandomForest::fit(Eigen::MatrixXd &features, Eigen::VectorXd &labels) {
  trees = std::vector<DecisionTree>(numberOfTrees);
  treesFeatureIndexes = std::vector<std::vector<unsigned int>>(numberOfTrees);

  for (unsigned int i = 0; i < numberOfTrees; i++) {
    std::cout << "Creating decision tree number: " << i << std::endl;

    trees.at(i) = DecisionTree(treeMaxDepth, treeMinSampleSplit, treeNumberOfFeatures);

    std::tuple<Eigen::MatrixXd, Eigen::VectorXd, std::vector<unsigned int>> bootstrapResult = bootstrap(features, labels);
    Eigen::VectorXd labelsSubset = std::get<1>(bootstrapResult);
    Eigen::MatrixXd featuresSubset = std::get<0>(bootstrapResult);
    treesFeatureIndexes.at(i) = std::get<2>(bootstrapResult);

    std::cout << "Started training decision tree number: " << i  << std::endl;

    trees.at(i).fit(featuresSubset, labelsSubset);

    std::cout << "Completed training decision tree number: " << i  << std::endl;
  }
}

Eigen::VectorXd RandomForest::predict(Eigen::MatrixXd &features) {
  std::cout << "Started predictions"  << std::endl;

  unsigned int numberOfDataSamples = (unsigned int) features.rows();

  Eigen::MatrixXd allPredictions(numberOfTrees, numberOfDataSamples);

  for (unsigned int i = 0; i < numberOfTrees; i++) {
    std::cout << "Started prediction for decision tree number: " << i  << std::endl;

    Eigen::MatrixXd featuresSubset(features.rows(), treesFeatureIndexes.at(i).size());

    for (unsigned int k = 0; k < treesFeatureIndexes.at(i).size(); k++) {
      featuresSubset.col(k) = features.col(treesFeatureIndexes.at(i).at(k));
    }

    Eigen::RowVectorXd predictions = trees.at(i).predict(featuresSubset);
    allPredictions.row(i) = predictions;

    std::cout << "Completed prediction for decision tree number: " << i  << std::endl;
  }

  Eigen::VectorXd results(numberOfDataSamples);

  std::cout << "Started final prediction"  << std::endl;

  for (unsigned int i = 0; i < numberOfDataSamples; i++) {
    Eigen::VectorXd predictions = allPredictions.col(i);
    
    double result = MLUtils::mostCommonLabel(predictions);

    results(i) = result;
  }

  std::cout << "Completed final prediction"  << std::endl;

  return results;
}
