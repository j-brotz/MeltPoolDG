#include <deal.II/base/exceptions.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_tools.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/vector.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/numerics/vector_tools_common.h>

#include <meltpooldg/utilities/amr_indicators.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>
#include <meltpooldg/utilities/matrix_free_util.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

#include <functional>
#include <utility>
