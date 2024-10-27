#include "RandomForest.hpp"

std::map<double, int> RandomForest::labelsFrequency(Eigen::VectorXd &labels) {
  std::map<double, int> frequency;

  for (int i = 0; i < labels.rows(); i++) {
    double label = labels(i);

    std::map<double, int>::iterator it = frequency.find(label);

    if (it == frequency.end()) {
      frequency.insert(std::pair<double, int>(label, 1));
    } else {
      it->second = it->second + 1;
    }
  }

  return frequency;
}

double RandomForest::mostCommonLabel(Eigen::VectorXd &labels) {
  std::map<double, int> labelsWithFrequency = labelsFrequency(labels);

  // get the label with highest frequency (majority vote)

  double mostCommonLabel = 0.0;
  int mostCommonLabelCount = 0;

  for (std::pair<double, int> label: labelsWithFrequency) {
     if (label.second > mostCommonLabelCount) {
      mostCommonLabel = label.first;
      mostCommonLabelCount = label.second;
    }
  }

  return mostCommonLabel;
}

std::tuple<Eigen::MatrixXd, Eigen::VectorXd> RandomForest::bootstrap(Eigen::MatrixXd &features, Eigen::VectorXd &labels) {
  unsigned int numberOfDataSamples = (unsigned int) features.rows();
  unsigned int numberOfDataFeatures = (unsigned int) features.cols();
  
  unsigned int numberOfSamplesToSelect = numberOfDataSamples <= 4 ? numberOfDataSamples : numberOfDataSamples - 2;
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

  return std::tuple<Eigen::MatrixXd, Eigen::VectorXd>(featuresSubset, labelsSubset);
}

void RandomForest::fit(Eigen::MatrixXd &features, Eigen::VectorXd &labels) {
  trees = std::vector<DecisionTree>(numberOfTrees);

  for (unsigned int i = 0; i < numberOfTrees; i++) {
    std::cout << "Creating decision tree number: " << i << std::endl;

    trees.at(i) = DecisionTree(3, treeMinSampleSplit, treeNumberOfFeatures);

    std::tuple<Eigen::MatrixXd, Eigen::VectorXd> bootstrapResult = bootstrap(features, labels);
    Eigen::VectorXd labelsSubset = std::get<1>(bootstrapResult);
    Eigen::MatrixXd featuresSubset = std::get<0>(bootstrapResult);

    std::cout << "Started training decision tree number: " << i  << std::endl;

    // tree.fit(featuresSubset, labelsSubset);
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

    Eigen::RowVectorXd predictions = trees.at(i).predict(features);
    allPredictions.row(i) = predictions;

    std::cout << "Completed prediction for decision tree number: " << i  << std::endl;
  }

  Eigen::VectorXd results(numberOfDataSamples);

  std::cout << "Started prediction final"  << std::endl;

  for (unsigned int i = 0; i < numberOfDataSamples; i++) {
    Eigen::VectorXd predictions = allPredictions.col(i);
    
    double result = mostCommonLabel(predictions);

    results(i) = result;
  }

  std::cout << "Completed prediction final"  << std::endl;

  return results;
}
