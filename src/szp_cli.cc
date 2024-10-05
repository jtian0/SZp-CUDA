#include <iostream>

#include "szp_demo.hh"

using std::cout;
using std::endl;

using namespace szp;

int main(int argc, char* argv[])
{
  if (argc < 7) {
    printf("     1      2  3  4  5   6\n");
    printf("szp  fname  x  y  z  eb  use_rel[yes|no]\n");
    exit(1);
  }

  auto const fname = std::string(argv[1]);
  auto const x = std::stoi(argv[2]);
  auto const y = std::stoi(argv[3]);
  auto const z = std::stoi(argv[4]);
  auto const eb = std::stod(argv[5]);
  auto const use_rel = std::string(argv[6]) == "yes";

  szp::compressor_roundtrip_float(fname, x, y, z, eb, use_rel);
  return 0;
}