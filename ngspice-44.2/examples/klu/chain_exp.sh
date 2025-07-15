export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
g++ -std=c++17 -o chain_exp chain_exp.cpp -I/usr/local/include/suitesparse/ -lumfpack -lamd -lcholmod -lklu  -lsuitesparseconfig -lm
  ./chain_exp c1355_cktmatrix.txt
  ./chain_exp c1908_cktmatrix.txt
  ./chain_exp c2670_cktmatrix.txt
  ./chain_exp c3540_cktmatrix.txt
 ./chain_exp c432_cktmatrix.txt
  ./chain_exp c499_cktmatrix.txt
  ./chain_exp c5315_cktmatrix.txt
  ./chain_exp c6288_cktmatrix.txt
  ./chain_exp c7552_cktmatrix.txt
  ./chain_exp c880_cktmatrix.txt

