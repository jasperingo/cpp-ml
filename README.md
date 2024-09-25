# CPP-ML

Implementations of machine learning algorithms.

## Implemented algorithms

- K-Nearest Neighbour (KNN)
- Linear Regression

## Dependencies

- Eigen (v3.3.9) [See here](https://eigen.tuxfamily.org/)
- matplotplusplus (v1.2.0) [See here](https://alandefreitas.github.io/matplotplusplus/)

## Usage

Clone project

`git clone https://github.com/jasperingo/cpp-ml.git`

Set up dependecies for CMake as describe in their respective documentations

Build project with CMake

Run executable

### Arguments

**--dataset**

Path to data set file

`--dataset /home/student-test-score.csv`

**--algorithm**

ML algorithm to used. Values can be any of:
- knn
- linear-regression 

`--algorithm knn`

**--label-column**

Index of the column in the dataset that contains the values for sample labels

Indexes start from 0

`--label-column 1`

**--label-column-digit**

Indicates if the value of the labels are digit or not. Values can be:
- true / 1
- false / 0 

`--label-column-digit true`

**--feature-columns** 

Comma separated indexes of the columns in the dataset that contains the values for sample features

Indexes start from 0

`--feature-columns 1,2,4`

**--feature-columns-digit**

Comma separated values that indicates if the value of the features are digit or not. Values can be:
- true / 1
- false / 0 

`--feature-columns-digit true,false,true`

**--learning-rate**

Optional argument

Used to specify a custom learning rate for algorithms that support it. Default value is 0.001

`--learning-rate 0.01`

**--max-iterations**

Optional argument

Used to specify a custom maximum number of iterations for algorithms that support it. Default value is 1000

`--max-iterations 2000`
