#include "DecisionTree.hpp"

void DecisionTree::deleteNodes(DecisionTree::Node* node) {
  if (!node->isLeafNode) {
    if (node->left != nullptr) {
      deleteNodes(node->left);
    }

    if (node->right != nullptr) {
      deleteNodes(node->right);
    }
  }

  delete node;
}

std::tuple<unsigned int, double> DecisionTree::bestSplit(Eigen::MatrixXd &features, Eigen::VectorXd &labels, std::vector<unsigned int> &featureIndexes) {
  double bestGain = -1.0;
  unsigned int splitIndex = -1;
  double splitThreshold = -1.0;

  for (unsigned int featureIndex: featureIndexes) {
    Eigen::VectorXd featureColumn = features.col(featureIndex);
    std::set<double> thresholds(featureColumn.data(), featureColumn.data() + featureColumn.size());

    for (double threshold: thresholds) {
      double gain = informationGain(labels, featureColumn, threshold);

      // std::cout << "Information gain for feature = " << featureIndex << " and threshold = " << threshold << " : " << gain << std::endl;

      if (gain > bestGain) {
        bestGain = gain;
        splitIndex = featureIndex;
        splitThreshold = threshold;
      }
    }
  }

  return std::tuple<unsigned int, double>(splitIndex, splitThreshold);
}

double DecisionTree::informationGain(Eigen::VectorXd &labels, Eigen::VectorXd &feature, double threshold) {
  double parentEntropy = entropy(labels);

  // std::cout << "Parent entropy: " << parentEntropy << std::endl;

  std::tuple<std::vector<unsigned int>, std::vector<unsigned int>> childrenIndexes = split(feature, threshold);
  std::vector<unsigned int> leftIndexes = std::get<0>(childrenIndexes);
  std::vector<unsigned int> rightIndexes = std::get<1>(childrenIndexes);

  if (leftIndexes.size() == 0 || rightIndexes.size() == 0) {
    return 0.0;
  }

  Eigen::VectorXd leftLabels(leftIndexes.size());
  Eigen::VectorXd rightLabels(rightIndexes.size());

  for (unsigned int i = 0; i < leftIndexes.size(); i++) {
    leftLabels(i) = labels(leftIndexes.at(i));
  }

  for (unsigned int i = 0; i < rightIndexes.size(); i++) {
    rightLabels(i) = labels(rightIndexes.at(i));
  }

  double leftEntropy = entropy(leftLabels);
  double rightEntropy = entropy(rightLabels);
  
  // std::cout << "Left child entropy: " << leftEntropy << std::endl;
  // std::cout << "Right child entropy: " << rightEntropy << std::endl;

  double childEntropy = 
    (((double) leftLabels.size() / (double) labels.size()) * leftEntropy) + 
    (((double) rightLabels.size() / (double) labels.size()) * rightEntropy);

  // std::cout << "Child entropy: " << childEntropy << std::endl;

  return parentEntropy - childEntropy;
}

double DecisionTree::entropy(Eigen::VectorXd &labels) {
  std::map<double, unsigned int> frequencyMap = MLUtils::labelsFrequency(labels);

  Eigen::VectorXd frequencies(frequencyMap.size());

  unsigned int i = 0;

  for (std::pair<double, unsigned int> item: frequencyMap) {
    frequencies(i) = (double) item.second;
    i++;
  }

  Eigen::ArrayXd pOfX = frequencies / labels.size();

  Eigen::ArrayXd pOfX2 = pOfX * pOfX.log();

  return -(pOfX2.sum());
}

std::tuple<std::vector<unsigned int>, std::vector<unsigned int>> DecisionTree::split(Eigen::VectorXd &feature, double threshold) {
  std::vector<unsigned int> leftIndexes;
  std::vector<unsigned int> rightIndexes;

  for (unsigned int i = 0; i < feature.size(); i++) {
    double value = feature(i);

    if (value <= threshold) {
      leftIndexes.push_back(i);
    } else {
      rightIndexes.push_back(i);
    }
  }

  return std::tuple<std::vector<unsigned int>, std::vector<unsigned int>>(leftIndexes, rightIndexes);
}

DecisionTree::Node* DecisionTree::grow(Eigen::MatrixXd &features, Eigen::VectorXd &labels, unsigned int depth) {
  std::cout << "Decision tree depth: " << depth << std::endl;

  unsigned int numberOfDataSamples = (unsigned int) features.rows();
  unsigned int numberOfDataFeatures = (unsigned int) features.cols();

  std::set<double> uniqueLabels(labels.data(), labels.data() + labels.size());

  if (depth >= maxDepth || uniqueLabels.size() == 1 || numberOfDataSamples < minSampleSplit) {
    DecisionTree::Node* node = new DecisionTree::Node;
    node->isLeafNode = true;
    node->value = MLUtils::mostCommonLabel(labels);
    return node;
  }

  unsigned int numberOfFoundFeatures = 0;

  std::vector<unsigned int> featureIndexes(numberOfFeatures);

  while (numberOfFoundFeatures < numberOfFeatures) {
    std::srand((unsigned int) std::time(0)); // use current time as seed for random generator
    int randomIndex = std::rand() % numberOfDataFeatures;  // Modulo to restrict the number of random values to be at most vector.size()-1
    std::vector<unsigned int>::iterator it = std::find(featureIndexes.begin(), featureIndexes.end(), randomIndex);

    if (it == featureIndexes.end()) {
      featureIndexes.at(numberOfFoundFeatures) = randomIndex;
      numberOfFoundFeatures++;
    }
  }

  std::tuple<unsigned int, double> bestSplitResult = bestSplit(features, labels, featureIndexes);

  unsigned int bestFeatureIndex = std::get<0>(bestSplitResult);
  double bestThreshold = std::get<1>(bestSplitResult);
  
  // std::cout << "Best feature index: " << bestFeatureIndex << " and best threshold: " << bestThreshold << std::endl;

  Eigen::VectorXd bestFeatureColumn = features.col(bestFeatureIndex);

  std::tuple<std::vector<unsigned int>, std::vector<unsigned int>> childrenIndexes = split(bestFeatureColumn, bestThreshold);
  std::vector<unsigned int> leftIndexes = std::get<0>(childrenIndexes);
  std::vector<unsigned int> rightIndexes = std::get<1>(childrenIndexes);

  Eigen::VectorXd leftLabels(leftIndexes.size());
  Eigen::MatrixXd leftFeatures(leftIndexes.size(), numberOfDataFeatures);

  Eigen::VectorXd rightLabels(rightIndexes.size());
  Eigen::MatrixXd rightFeatures(rightIndexes.size(), numberOfDataFeatures);

  for (unsigned int i = 0; i < leftIndexes.size(); i++) {
    leftLabels(i) = labels(leftIndexes.at(i));
    leftFeatures.row(i) = features.row(leftIndexes.at(i));
  }

  for (unsigned int i = 0; i < rightIndexes.size(); i++) {
    rightLabels(i) = labels(rightIndexes.at(i));
    rightFeatures.row(i) = features.row(rightIndexes.at(i));
  }

  DecisionTree::Node* leftNode = grow(leftFeatures, leftLabels, depth + 1);
  DecisionTree::Node* rightNode = grow(rightFeatures, rightLabels, depth + 1);

  DecisionTree::Node* resultNode = new DecisionTree::Node;
  resultNode->isLeafNode = false;
  resultNode->left = leftNode;
  resultNode->right = rightNode;
  resultNode->threshold = bestThreshold;
  resultNode->feature = bestFeatureIndex;

  return resultNode;
}

double DecisionTree::traverse(Eigen::RowVectorXd &sample, DecisionTree::Node *node) {
  if (node->isLeafNode) {
    return node->value;
  }

  if (sample(node->feature) <= node->threshold) {
    return traverse(sample, node->left);
  }

  return traverse(sample, node->right);
}

void DecisionTree::fit(Eigen::MatrixXd &features, Eigen::VectorXd &labels) {
  unsigned int numberOfDataFeatures = (unsigned int) features.cols();

  numberOfFeatures = numberOfFeatures > 0 && numberOfFeatures < numberOfDataFeatures
    ? numberOfFeatures
    : numberOfDataFeatures;

  rootNode = grow(features, labels);
}

Eigen::VectorXd DecisionTree::predict(Eigen::MatrixXd &features) {
  Eigen::VectorXd result(features.rows());

  for (unsigned int i = 0; i < features.rows(); i++) {
    Eigen::RowVectorXd sample = features.row(i); 
    result(i) = traverse(sample, rootNode);
  }

  return result;
}
