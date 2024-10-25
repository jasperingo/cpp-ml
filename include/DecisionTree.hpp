#ifndef CPP_ML_DECISION_TREE_H_
#define CPP_ML_DECISION_TREE_H_

#include <iostream>
#include <map>
#include <set>
#include <vector>
#include <cmath>
#include <ctime>
#include <Eigen/Dense>

class DecisionTree {
  struct Node {
    double value;
    bool isLeafNode;
    double threshold;
    unsigned int feature;
    DecisionTree::Node* left;
    DecisionTree::Node* right;
  };

  unsigned int maxDepth;
  unsigned int minSampleSplit;
  unsigned int numberOfFeatures;
  DecisionTree::Node* rootNode;

  void deleteNodes(DecisionTree::Node* node);

  std::map<double, int> labelsFrequency(Eigen::VectorXd& labels);

  double mostCommonLabel(Eigen::VectorXd& labels);

  std::tuple<unsigned int, double> bestSplit(Eigen::MatrixXd& features, Eigen::VectorXd& labels, std::vector<unsigned int>& featureIndexes);

  double informationGain(Eigen::VectorXd& labels, Eigen::VectorXd& feature, double threshold);

  double entropy(Eigen::VectorXd& labels);

  std::tuple<std::vector<unsigned int>, std::vector<unsigned int>> split(Eigen::VectorXd& feature, double threshold);

  DecisionTree::Node* grow(Eigen::MatrixXd& features, Eigen::VectorXd& labels, unsigned int depth = 0);

  double traverse(Eigen::RowVectorXd& sample, DecisionTree::Node* node);

public:

  DecisionTree(unsigned int maxDepth, unsigned int minSampleSplit, unsigned int numberOfFeatures) : 
    maxDepth(maxDepth), 
    minSampleSplit(minSampleSplit), 
    numberOfFeatures(numberOfFeatures) {}
  
  ~DecisionTree() {
    deleteNodes(rootNode);
  }

  void fit(Eigen::MatrixXd& features, Eigen::VectorXd& labels);

  Eigen::VectorXd predict(Eigen::MatrixXd& features);
};

#endif /* CPP_ML_DECISION_TREE_H_ */
