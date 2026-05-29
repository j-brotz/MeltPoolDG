#pragma once

#include <deal.II/base/point.h>
#include <deal.II/base/vectorization.h>

#include <meltpooldg/compressible_flow/data_types.hpp>
#include <meltpooldg/particles/obstacle_field.hpp>

#include <memory>
#include <utility>

namespace MeltPoolDG::FSI
{
  template <int dim, typename number, typename ObstacleType>
  class PenaltyCouplingTypeErasure
  {
    using ConservedVariablesType = CompressibleFlow::ConservedVariablesType<dim, number>;

  public:
    PenaltyCouplingTypeErasure() = default;

    template <typename T>
    PenaltyCouplingTypeErasure(std::unique_ptr<T> &&model)
      : penalty_coupling_pimpl(std::make_unique<PenaltyCouplingModel<T>>(std::move(model)))
    {}

    void
    add_load_to_obstacles(ObstacleField<dim, number, ObstacleType> &obstacle_field) const
    {
      penalty_coupling_pimpl->add_load_to_obstacles(obstacle_field);
    }

    ConservedVariablesType
    fluid_penalty_force(number                                                     time_step_size,
                        const unsigned int                                         cell_batch_id,
                        const dealii::Point<dim, dealii::VectorizedArray<number>> &points,
                        const ConservedVariablesType                              &w) const
    {
      return penalty_coupling_pimpl->fluid_penalty_force(time_step_size, cell_batch_id, points, w);
    }

    ConservedVariablesType
    fluid_penalty_force_jacobian(number             time_step_size,
                                 const unsigned int cell_batch_id,
                                 const dealii::Point<dim, dealii::VectorizedArray<number>> &points,
                                 const ConservedVariablesType                              &w,
                                 const ConservedVariablesType &delta_w) const
    {
      return penalty_coupling_pimpl->fluid_penalty_force_jacobian(
        time_step_size, cell_batch_id, points, w, delta_w);
    }

  private:
    class PenaltyCouplingConcept
    {
    public:
      virtual ~PenaltyCouplingConcept() = default;

      virtual void
      add_load_to_obstacles(ObstacleField<dim, number, ObstacleType> &obstacle_field) const = 0;

      virtual ConservedVariablesType
      fluid_penalty_force(number                                                     time_step_size,
                          const unsigned int                                         cell_batch_id,
                          const dealii::Point<dim, dealii::VectorizedArray<number>> &q_point,
                          const ConservedVariablesType                              &w_q) = 0;

      virtual ConservedVariablesType
      fluid_penalty_force_jacobian(
        number                                                     time_step_size,
        const unsigned int                                         cell_batch_id,
        const dealii::Point<dim, dealii::VectorizedArray<number>> &q_point,
        const ConservedVariablesType                              &w_q,
        const ConservedVariablesType                              &delta_w_q) = 0;
    };

    template <typename T>
    class PenaltyCouplingModel : public PenaltyCouplingConcept
    {
    public:
      PenaltyCouplingModel(std::unique_ptr<T> &&model)
        : model(std::move(model))
      {}

      void
      add_load_to_obstacles(ObstacleField<dim, number, ObstacleType> &obstacle_field) const override
      {
        model->add_load_to_obstacles(obstacle_field);
      }

      ConservedVariablesType
      fluid_penalty_force(number                                                     time_step_size,
                          const unsigned int                                         cell_batch_id,
                          const dealii::Point<dim, dealii::VectorizedArray<number>> &q_point,
                          const ConservedVariablesType                              &w_q) override
      {
        return model->fluid_penalty_force(time_step_size, cell_batch_id, q_point, w_q);
      }

      ConservedVariablesType
      fluid_penalty_force_jacobian(
        number                                                     time_step_size,
        const unsigned int                                         cell_batch_id,
        const dealii::Point<dim, dealii::VectorizedArray<number>> &q_point,
        const ConservedVariablesType                              &w_q,
        const ConservedVariablesType                              &delta_w_q) override
      {
        return model->fluid_penalty_force_jacobian(
          time_step_size, cell_batch_id, q_point, w_q, delta_w_q);
      }

    private:
      std::unique_ptr<T> model;
    };

    std::unique_ptr<PenaltyCouplingConcept> penalty_coupling_pimpl;
  };
} // namespace MeltPoolDG::FSI