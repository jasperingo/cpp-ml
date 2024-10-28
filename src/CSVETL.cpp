#include "CSVETL.hpp"

bool CSVETL::loadFile() {
  std::ifstream csvFile(filePath);

  if (!csvFile.is_open()) {
    std::cout << "Dataset file could not be opened" << std::endl;
    return false;
  }

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

  return true;
}

bool CSVETL::extractLabels(unsigned int labelColumn, bool labelColumIsDigit) {
  size_t colSize = data[0].size();

  if (colSize <= labelColumn) {
    std::cout << "Label column: " << labelColumn << " exceeds the column indexes of " << colSize << " in the provided dataset" << std::endl;
    return false;
  }

  size_t rowSize = data.size();

  Eigen::VectorXd vector(rowSize);

  std::vector<std::string> noneDigitColumnValues;

  for (int i = 0; i < rowSize; i++) {
    std::string columnValue = data[i][labelColumn];

    if (labelColumIsDigit) {
      vector(i) = std::atof(columnValue.c_str());
    } else {
      size_t column;
      bool columnFound = false;

      for (size_t k = 0; k < noneDigitColumnValues.size(); k++) {
        if (columnValue == noneDigitColumnValues[k]) {
          column = k;
          columnFound = true;
          break;
        }
      }

      if (!columnFound) {
        column = noneDigitColumnValues.size();
        noneDigitColumnValues.push_back(columnValue);
      }

      vector(i) = (double) column;
    }
  }

  labels = vector;

  return true;
}

bool CSVETL::extractFeatures(std::vector<unsigned int> featureColumns, std::vector<bool> featureColumnsAreDigits) {
  size_t colSize = data[0].size();

  for (size_t i = 0; i < featureColumns.size(); i++) {
    if (colSize <= featureColumns[i]) {
      std::cout << "Feature column: " << featureColumns[i] << " exceeds the column indexes of " << colSize << " in the provided dataset" << std::endl;
      return false;
    }
  }

  size_t rowSize = data.size();

  Eigen::MatrixXd matrix(rowSize, featureColumns.size());

  std::map<int, std::vector<std::string>> noneDigitColumnValues;

  for (int i = 0; i < rowSize; i++) {
    for (int j = 0; j < featureColumns.size(); j++) {
      int columnIndex = featureColumns[j];
      bool columnIsDigit = featureColumnsAreDigits[j];
      std::string columnValue = data[i][columnIndex];

      if (columnIsDigit) {
        matrix(i, j) = std::atof(columnValue.c_str());
      } else {
        std::map<int, std::vector<std::string>>::iterator itr = noneDigitColumnValues.find(columnIndex);

        if (itr == noneDigitColumnValues.end()) {
          std::vector<std::string> noneDigitColumnValue;
          noneDigitColumnValue.push_back(columnValue);
          noneDigitColumnValues.insert(std::pair<int, std::vector<std::string>>(columnIndex, noneDigitColumnValue));
          matrix(i, j) = 0.0;
        } else {
          size_t column;
          bool columnFound = false;

          for (size_t k = 0; k < itr->second.size(); k++) {
            if (columnValue == itr->second[k]) {
              column = k;
              columnFound = true;
              break;
            }
          }

          if (!columnFound) {
            column = itr->second.size();
            itr->second.push_back(columnValue);
          }

          matrix(i, j) = (double) column;
        }
      }
    }
  }

  features = matrix;

  return true;
}

bool CSVETL::load(
  unsigned int labelColumn, 
  bool labelColumIsDigit, 
  std::vector<unsigned int> featureColumns, 
  std::vector<bool> featureColumnsAreDigits
) {

  if (!loadFile()) {
    return false;
  }

  std::cout << "Loaded: " << data.size() << " rows of data" << std::endl;

  if (data.size() == 0) {
    return false;
  }

  if (!extractLabels(labelColumn, labelColumIsDigit)) {
    return false;
  }

  if (!extractFeatures(featureColumns, featureColumnsAreDigits)) {
    return false;
  }

  return true;
}

CSVETL::DataSplitResult CSVETL::splitData(float testSize) {
  size_t dataSize = labels.rows();

  std::vector<size_t> indexes(dataSize);

  std::iota(indexes.begin(), indexes.end(), 0);

  std::shuffle(indexes.begin(), indexes.end(), std::random_device());

  size_t testDataSize = (size_t) (dataSize * testSize);

  size_t trainDataSize = dataSize - testDataSize;

  Eigen::VectorXd trainLabels(trainDataSize);
  Eigen::VectorXd testLabels(testDataSize);
  Eigen::MatrixXd trainSamples(trainDataSize, features.cols());
  Eigen::MatrixXd testSamples(testDataSize, features.cols());

  for (size_t i = 0; i < trainDataSize; i++) {
    size_t index = indexes[i];
    trainLabels.row(i) = labels.row(index);
    trainSamples.row(i) = features.row(index);
  }

  for (size_t i = 0; i < testDataSize; i++) {
    size_t index = indexes[i + trainDataSize];
    testLabels.row(i) = labels.row(index);
    testSamples.row(i) = features.row(index);
  }

  DataSplitResult result;
  result.testLabels = testLabels;
  result.trainLabels = trainLabels;
  result.testSamples = testSamples;
  result.trainSamples = trainSamples;
  
  return result;
}

void CSVETL::printDataset() {
  std::cout << "Features:" << std::endl;

  std::cout << features << std::endl << std::endl;

  std::cout << "Labels:" << std::endl;

  std::cout << labels << std::endl << std::endl;
}

void CSVETL::printDatasetSize() {
  std::cout << "Number of samples: " << features.rows() << std::endl;
  std::cout << "Number of features: " << features.cols() << std::endl;
  std::cout << "Number of labels: " << labels.size() << std::endl;
}

void CSVETL::printDatasetSplitSize(DataSplitResult split) {
  std::cout << "Number of train samples: " << split.trainSamples.rows() << std::endl;
  std::cout << "Number of train labels: " << split.trainLabels.size() << std::endl;
  
  std::cout << "Number of test samples: " << split.testSamples.rows() << std::endl;
  std::cout << "Number of test labels: " << split.testLabels.size() << std::endl;
}
