#ifndef CPP_ML_DECISION_TREE_H_
#define CPP_ML_DECISION_TREE_H_

#include <Eigen/Dense>

class DecisionTree {
  struct Node {
    double feature;
    double threshold;
    DecisionTree::Node* right;
    DecisionTree::Node* left;
    double value;
    bool isLeafNode;
  };

  unsigned int maxDepth;
  unsigned int minSampleSplit;
  unsigned int numberOfFeatures;
  DecisionTree::Node root;

public:

  DecisionTree( unsigned int maxDepth, unsigned int minSampleSplit, unsigned int numberOfFeatures) : 
    maxDepth(maxDepth), 
    minSampleSplit(minSampleSplit), 
    numberOfFeatures(numberOfFeatures) {}

  void fit(Eigen::MatrixXd& features, Eigen::VectorXd& labels);

  Eigen::VectorXd predict(Eigen::MatrixXd& fetures);
};

#endif /* CPP_ML_DECISION_TREE_H_ */
