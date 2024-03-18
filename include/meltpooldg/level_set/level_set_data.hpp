#pragma once

#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/advection_diffusion/advection_diffusion_data.hpp>
#include <meltpooldg/curvature/curvature_data.hpp>
#include <meltpooldg/interface/finite_element_data.hpp>
#include <meltpooldg/level_set/nearest_point_data.hpp>
#include <meltpooldg/normal_vector/normal_vector_data.hpp>
#include <meltpooldg/reinitialization/reinitialization_data.hpp>

namespace MeltPoolDG::LevelSet
{

  template <typename number = double>
  struct LevelSetData
  {
    FiniteElementData fe;

    bool do_localized_heaviside = true;

    NearestPointData<number> nearest_point;

    AdvectionDiffusionData<number> advec_diff;
    NormalVectorData<number>       normal_vec;
    CurvatureData<number>          curv;
    ReinitializationData<number>   reinit;

    void
    add_parameters(dealii::ParameterHandler &prm);

    void
    post(const FiniteElementData &base_fe_data);

    void
    check_input_parameters(const FiniteElementData &base_fe_data) const;

    unsigned int
    get_n_subdivisions() const;
  };
} // namespace MeltPoolDG::LevelSet
