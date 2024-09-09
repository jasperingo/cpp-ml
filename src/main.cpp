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

  if (csvETL.load()) {
    std::cout << "Dataset: " << argv[1] << " loaded " << std::endl;
  }

  // KNN knn;

  // std::cout << "Na me da here!! " << knn.test() << std::endl;

  // Eigen::MatrixXd m(2,2);
  // m(0,0) = 3;
  // m(1,0) = 2.1;
  // m(0,1) = -2;
  // m(1,1) = m(1,0) + m(0,1);

  // std::cout << m << std::endl;
}
