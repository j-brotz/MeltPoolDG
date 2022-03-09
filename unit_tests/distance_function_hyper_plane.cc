#include <deal.II/base/point.h>

#include <meltpooldg/utilities/distance_functions.hpp>

#include <iostream>

using namespace dealii;
using namespace MeltPoolDG::DistanceFunctions;

int
main()
{
  std::cout.precision(4);
  {
    std::cout << "dim 1" << std::endl;
    const Point<1> supp(1.0);
    const Point<1> norm(-1.0);
    std::cout << "distance: " << hyper_plane(supp, supp, norm) << std::endl;
    std::cout << "distance: " << hyper_plane(Point<1>(0.0), supp, norm) << std::endl;
    std::cout << "distance: " << hyper_plane(Point<1>(2.5), supp, norm) << std::endl;
  }
  {
    std::cout << "dim 2" << std::endl;
    const Point<2> supp(1.0, 0.0);
    const Point<2> norm(-1.0, -1.0);
    std::cout << "distance: " << hyper_plane(supp, supp, norm) << std::endl;
    std::cout << "distance: " << hyper_plane(Point<2>(0.0, 0.0), supp, norm) << std::endl;
    std::cout << "distance: " << hyper_plane(Point<2>(2.0, 1.0), supp, norm) << std::endl;
  }
  {
    std::cout << "dim 3" << std::endl;
    const Point<3> supp(1.0, 0.0, 0.0);
    const Point<3> norm(1.0, 1.0, 1.0);
    std::cout << "distance: " << hyper_plane(supp, supp, norm) << std::endl;
    std::cout << "distance: " << hyper_plane(Point<3>(0.0, 0.0, 0.0), supp, norm) << std::endl;
    std::cout << "distance: " << hyper_plane(Point<3>(2.0, 1.0, 1.0), supp, norm) << std::endl;

    // three support points
    const Point<3> A(0.0, 0.0, 0.0);
    const Point<3> B(1.0, 0.0, 0.0);
    const Point<3> C(0.0, 1.0, 0.0);
    std::cout << "distance: " << hyper_plane(Point<3>(2.0, 2.0, 1.0), A, B, C) << std::endl;
    std::cout << "distance: " << hyper_plane(Point<3>(-2.0, -2.0, -2.0), A, B, C) << std::endl;
  }
}
