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
  if (data[0].size() >= labelColumn) {
    std::cout << "Label column: " << labelColumn << " exceeds the column indexes in the provided dataset" << std::endl;
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

bool CSVETL::extractSamples(unsigned int sampleColumns[], bool sampleColumnsAreDigits[], size_t sampleColumnsLength) {
  size_t colSize = data[0].size();

  for (size_t i = 0; i < sampleColumnsLength; i++) {
    if (colSize >= sampleColumns[i]) {
      std::cout << "Sample column: " << sampleColumns[i] << " exceeds the column indexes in the provided dataset" << std::endl;
      return false;
    }
  }

  size_t rowSize = data.size();

  Eigen::MatrixXd matrix(rowSize, sampleColumnsLength);

  std::map<int, std::vector<std::string>> noneDigitColumnValues;

  for (int i = 0; i < rowSize; i++) {
    for (int j = 0; j < sampleColumnsLength; j++) {
      int columnIndex = sampleColumns[j];
      bool columnIsDigit = sampleColumnsAreDigits[j];
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

  return true;
}

bool CSVETL::load(
  unsigned int labelColumn, 
  bool labelColumIsDigit, 
  unsigned int sampleColumns[], 
  bool sampleColumnsAreDigits[], 
  size_t sampleColumnsLength
) {

  if (!loadFile()) {
    return false;
  }

  std::cout << "Loaded: " << data.size() << " rows of data" << std::endl;

  if (!extractLabels(labelColumn, labelColumIsDigit)) {
    return false;
  }

  if (!extractSamples(sampleColumns, sampleColumnsAreDigits, sampleColumnsLength)) {
    return false;
  }

  return true;
}

CSVETL::DataSplitResult CSVETL::splitData(Eigen::VectorXd& labels, Eigen::MatrixXd& samples, float testSize) {
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
