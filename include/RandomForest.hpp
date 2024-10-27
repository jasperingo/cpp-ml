#ifndef CPP_ML_RANDOM_FOREST_H_
#define CPP_ML_RANDOM_FOREST_H_

#include <map>
#include <set>
#include <vector>
#include <ctime>
#include <iostream>
#include <Eigen/Dense>
#include "DecisionTree.hpp"

class RandomForest {
  unsigned int treeMaxDepth;
  unsigned int treeMinSampleSplit;
  unsigned int treeNumberOfFeatures;
  unsigned int numberOfTrees;
  std::vector<DecisionTree> trees;

  std::map<double, int> labelsFrequency(Eigen::VectorXd& labels);

  double mostCommonLabel(Eigen::VectorXd& labels);

  std::tuple<Eigen::MatrixXd, Eigen::VectorXd> bootstrap(Eigen::MatrixXd& features, Eigen::VectorXd& labels);

public:

  RandomForest(unsigned int numberOfTrees, unsigned int treeMaxDepth, unsigned int treeMinSampleSplit, unsigned int treeNumberOfFeatures) : 
    numberOfTrees(numberOfTrees), 
    treeMaxDepth(treeMaxDepth), 
    treeMinSampleSplit(treeMinSampleSplit), 
    treeNumberOfFeatures(treeNumberOfFeatures) {}

  void fit(Eigen::MatrixXd& features, Eigen::VectorXd& labels);

  Eigen::VectorXd predict(Eigen::MatrixXd& features);
};

#endif /* CPP_ML_RANDOM_FOREST_H_ */
