#include "CSVETL.hpp"

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
