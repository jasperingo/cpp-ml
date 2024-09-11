#include "CSVETL.hpp"

int CSVETL::findIndexInColumns(size_t column, size_t columns[], size_t columnsLength) {
  for (int i = 0; i < columnsLength; i++) {
    if (columns[i] == column) {
      return i;
    }
  }

  return -1;
}

bool CSVETL::load() {
  std::ifstream csvFile(filePath);

  if (!csvFile.is_open()) {
    std::cout << "File was not opened" << std::endl;
    return false;
  }

  std::cout << "File was opened" << std::endl;

  std::string fileLine;
  
  bool firstLineRead = false;
  
  while (std::getline(csvFile, fileLine)) {
    if (!firstLineRead) {
      firstLineRead = true;
      continue;
    }

    std::string linePart;
    std::vector<std::string> row;
    std::stringstream lineStream(fileLine);

    while(std::getline(lineStream, linePart, delimiter)) {
      row.push_back(linePart);
    }

    data.push_back(row);
  }
  
  csvFile.close();

  std::cout << "Loaded: " << data.size() << " rows of data" << std::endl;

  return true;
}

Eigen::MatrixXd CSVETL::extractSamples(size_t sampleColumns[], size_t sampleColumnsLength) {
  if (data.size() == 0 || data[0].size() == 0) {
    return Eigen::MatrixXd();
  }

  size_t rowSize = data.size();

  size_t colSize = data[0].size();

  Eigen::MatrixXd matrix(rowSize, sampleColumnsLength);

  for (int i = 0; i < rowSize; i++) {
    for (int j = 0; j < colSize; j++) {
      int columnIndex = findIndexInColumns(j, sampleColumns, sampleColumnsLength);

      if (columnIndex == -1) {
        continue;
      }

      matrix(i, columnIndex) = std::atof(data[i][j].c_str());
    }
  }

  return matrix;
}

Eigen::VectorXd CSVETL::extractLabels(size_t labelColumn) {
  if (data.size() == 0 || data[0].size() == 0) {
    return Eigen::VectorXd();
  }

  size_t rowSize = data.size();

  size_t colSize = data[0].size();

  Eigen::VectorXd vector(rowSize);

  std::vector<std::string> columnValues;

  for (int i = 0; i < rowSize; i++) {
    for (int j = 0; j < colSize; j++) {
      if (j != labelColumn) {
        continue;
      }

      size_t column;

      std::string columnValue = data[i][j];

      bool columnFound = false;

      for (size_t k = 0; k < columnValues.size(); k++) {
        if (columnValue == columnValues[k]) {
          column = k;
          columnFound = true;
          break;
        }
      }

      if (!columnFound) {
        column = columnValues.size();
        columnValues.push_back(columnValue);
      }

      vector(i) = (double) column;
    }
  }

  return vector;
}

CSVETL::DataSplitResult CSVETL::splitData(Eigen::VectorXd labels, Eigen::MatrixXd samples, float testSize) {
  size_t dataSize = labels.rows();

  std::vector<size_t> indexes(dataSize);

  std::iota(indexes.begin(), indexes.end(), 0);

  std::shuffle(indexes.begin(), indexes.end(), std::random_device());

  size_t testDataSize = (size_t) (dataSize * testSize);

  size_t trainDataSize = dataSize - testDataSize;

  Eigen::VectorXd trainLabels(trainDataSize);
  Eigen::VectorXd testLabels(testDataSize);
  Eigen::MatrixXd trainSamples(trainDataSize, samples.cols());
  Eigen::MatrixXd testSamples(testDataSize, samples.cols());

  for (size_t i = 0; i < trainDataSize; i++) {
    size_t index = indexes[i];
    trainLabels.row(i) = labels.row(index);
    trainSamples.row(i) = samples.row(index);
  }

  for (size_t i = 0; i < testDataSize; i++) {
    size_t index = indexes[i + trainDataSize];
    testLabels.row(i) = labels.row(index);
    testSamples.row(i) = samples.row(index);
  }

  DataSplitResult result;
  
  result.testLabels = testLabels;
  result.trainLabels = trainLabels;
  result.testSamples = testSamples;
  result.trainSamples = trainSamples;
  
  return result;
}
