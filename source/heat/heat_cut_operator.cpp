#include <meltpooldg/heat/heat_cut_operator.hpp>
//

#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>
#include <deal.II/base/utilities.h>

#include <deal.II/fe/fe_update_flags.h>

#include <deal.II/matrix_free/fe_evaluation_data.h>
#include <deal.II/matrix_free/fe_point_evaluation.h>
#include <deal.II/matrix_free/tools.h>

#include <meltpooldg/core/exceptions.hpp>
#include <meltpooldg/utilities/cut_util.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

#include <functional>

namespace MeltPoolDG::Heat
{
  template <int dim, typename number, int n_components = 1>
  using DomainEval = dealii::FECellIntegrator<dim, n_components, number>;
  template <int dim, typename number, int n_components = 1>
  using PointEval =
    dealii::FEPointEvaluation<n_components, dim, dim, dealii::VectorizedArray<number>>;
  template <int dim, typename number>
  using FaceEval = FEFaceIntegrator<dim, 1, number>;


  namespace internal
  {
    template <int dim, typename number>
    dealii::VectorizedArray<number>
    evaluate_function(const dealii::Function<dim, number>                       &function,
                      const dealii::Point<dim, dealii::VectorizedArray<number>> &p_vectorized,
                      const unsigned int                                         component = 0)
    {
      dealii::VectorizedArray<number> result;
      for (unsigned int v = 0; v < dealii::VectorizedArray<number>::size(); ++v)
        {
          dealii::Point<dim> p;
          for (unsigned int d = 0; d < dim; ++d)
            p[d] = p_vectorized[d][v];
          result[v] = function.value(p, component);
        }
      return result;
    }

    /*
     * vectorized implementation of the function in meltpooldg/heat/intensity_profiles.hpp with the
     * same name
     */
    template <int dim, typename number>
    inline dealii::VectorizedArray<number>
    compute_projection_factor(
      const dealii::Tensor<1, dim, number>                          &laser_direction,
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> &normal_vector)
    {
      Assert(std::abs(laser_direction.norm() - 1.0) < 1e-8,
             ExcMessage("The laser direction must be a unit vector"));
      // Assert(std::abs(normal_vector.norm() - 1.0) < 1e-8,
      //        ExcMessage("The laser direction must be a unit vector"));

      const auto fac = normal_vector * laser_direction;
      return dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(fac, 0.0, 0.0, fac);
    }

    template <int dim, typename number>
    inline dealii::VectorizedArray<number>
    compute_laser_heat_source(const dealii::Function<dim, number>  *laser_intensity_profile,
                              const dealii::Tensor<1, dim, number> &laser_direction,
                              const PointEval<dim, number>         &temp_eval,
                              const unsigned int                    q)
    {
      if (laser_intensity_profile == nullptr)
        return dealii::VectorizedArray<number>(0.0);

      return internal::evaluate_function<dim, number>(*laser_intensity_profile,
                                                      temp_eval.real_point(q)) *
             internal::compute_projection_factor(laser_direction, -temp_eval.normal_vector(q));
    }

    template <typename number>
    dealii::VectorizedArray<number>
    compute_qVapor(const dealii::VectorizedArray<number> &T)
    {
      DEAL_II_NOT_IMPLEMENTED();
      (void)T;
      return dealii::VectorizedArray<number>(0.0);
    }

    template <typename number>
    dealii::VectorizedArray<number>
    compute_qVapor_derivative(const dealii::VectorizedArray<number> &T)
    {
      DEAL_II_NOT_IMPLEMENTED();
      (void)T;
      return dealii::VectorizedArray<number>(0.0);
    }

    template <typename Evaluation>
    inline void
    set_dof_values_zero(Evaluation &eval)
    {
      typename Evaluation::value_type *dof_values = eval.begin_dof_values();
      for (const unsigned int j : eval.dof_indices())
        dof_values[j] = typename Evaluation::value_type();
    }
  } // namespace internal

  template <int dim, typename number>
  HeatCutOperator<dim, number>::HeatCutOperator(
    const ScratchData<dim>                     &scratch_data_in,
    const HeatData<number>                     &heat_data_in,
    const MaterialData<number>                 &material_data_in,
    const Evaporation::EvaporationData<number> &evapor_data_in,
    const unsigned int                          temp_dof_idx_in,
    const unsigned int                          temp_hanging_nodes_dof_idx_in,
    const unsigned int                          temp_quad_idx_in,
    const VectorType                           &temperature_in,
    dealii::NonMatching::MappingInfo<dim, dim, dealii::VectorizedArray<number>>
      &mapping_info_surface_in,
    const std::vector<
      std::shared_ptr<dealii::NonMatching::MappingInfo<dim, dim, dealii::VectorizedArray<number>>>>
                      &mapping_info_cells_in,
    const bool         do_solidification_in,
    const unsigned int vel_dof_idx_in,
    const VectorType  *velocity_in)
    : scratch_data(scratch_data_in)
    , heat_data(heat_data_in)
    , material_data(material_data_in)
    , evapor_data(evapor_data_in)
    , temp_dof_idx(temp_dof_idx_in)
    , temp_hanging_nodes_dof_idx(temp_hanging_nodes_dof_idx_in)
    , temp_quad_idx(temp_quad_idx_in)
    , temperature(temperature_in)
    , mapping_info_surface(mapping_info_surface_in)
    , mapping_info_cells(mapping_info_cells_in)
    , fe_point_temp(heat_data.fe.degree)
    , n_dofs_per_cell(fe_point_temp.dofs_per_cell)
    , fe_point_vel(dealii::FE_DGQ<dim>(heat_data.fe.degree), dim)
    , n_dofs_per_cell_vel(fe_point_vel.dofs_per_cell)
    , kappa_l(material_data.gas.thermal_conductivity /
              (material_data.gas.thermal_conductivity + material_data.liquid.thermal_conductivity))
    , kappa_g(material_data.liquid.thermal_conductivity /
              (material_data.gas.thermal_conductivity + material_data.liquid.thermal_conductivity))
    , evaluation_flags_surface(heat_data.cut.two_phase ? dealii::EvaluationFlags::values |
                                                           dealii::EvaluationFlags::gradients :
                                                         dealii::EvaluationFlags::values)
    , vel_dof_idx(vel_dof_idx_in)
    , velocity(velocity_in)
    , do_solidification(do_solidification_in)
  {
    // TODO external heat source

    AssertThrow(heat_data.linear_solver.do_matrix_free, dealii::ExcNotImplemented());
    AssertThrow(heat_data.fe.type == FiniteElementType::FE_Q,
                dealii::ExcMessage("only standard FE_Q elements are supported for now"));
    AssertThrow(heat_data.fe.degree == 1, dealii::ExcMessage("only degree 1 is supported for now"));

    // TODO material assertion, see heat diffuse operator
  }



  template <int dim, typename number>
  void
  HeatCutOperator<dim, number>::register_laser_intensity_function_and_direction(
    std::shared_ptr<const dealii::Function<dim, number>> laser_intensity_profile_in,
    const dealii::Tensor<1, dim, number>                &laser_direction_in)
  {
    laser_intensity_profile = laser_intensity_profile_in;
    laser_direction         = laser_direction_in;
  }



  template <int dim, typename number>
  void
  HeatCutOperator<dim, number>::reinit()
  {
    // precompute cell size (constant cell size for homogeneous Cartesian mesh)
    cell_side_length = scratch_data.get_min_cell_size();

    // precompute Nitsche term factor for the two-phase case
    if (heat_data.cut.two_phase)
      weighted_nitsche_factor =
        heat_data.cut.nitsche_parameter *
        (material_data.liquid.thermal_conductivity * material_data.gas.thermal_conductivity) /
        (material_data.liquid.thermal_conductivity + material_data.gas.thermal_conductivity) /
        cell_side_length;
  }



  template <int dim, typename number>
  void
  HeatCutOperator<dim, number>::init_time_advance(const double dt)
  {
    AssertThrowZeroTimeIncrement(dt);
    this->reset_time_increment(dt);
    ost_factor_implicit = this->time_increment * heat_data.cut.theta;
    ost_factor_explicit = this->time_increment * (1. - heat_data.cut.theta);
    if (heat_data.cut.two_phase)
      nitsche_factor = this->time_increment * weighted_nitsche_factor;
  }



  template <int dim, typename number>
  void
  HeatCutOperator<dim, number>::update_ghost_values() const
  {
    if (velocity and not velocity->has_ghost_elements())
      velocity->update_ghost_values();
  }



  template <int dim, typename number>
  void
  HeatCutOperator<dim, number>::vmult(VectorType &dst, const VectorType &src) const
  {
    // TODO dst.zero_out_ghost_values();

    scratch_data.get_matrix_free().loop(
      &HeatCutOperator::local_apply_domain_tangent,
      &HeatCutOperator::local_apply_inner_face_tangent,
      &HeatCutOperator::local_apply_boundary_face_tangent,
      this,
      dst,
      src,
      true,
      dealii::MatrixFree<dim, number, dealii::VectorizedArray<number>>::DataAccessOnFaces::
        gradients,
      dealii::MatrixFree<dim, number, dealii::VectorizedArray<number>>::DataAccessOnFaces::
        gradients);
  }



  template <int dim, typename number>
  void
  HeatCutOperator<dim, number>::create_rhs(VectorType       &residual,
                                           const VectorType &temperature_old) const
  {
    // TODO residual.zero_out_ghost_values();

    scratch_data.get_matrix_free().loop(
      &HeatCutOperator::local_apply_domain_residual,
      &HeatCutOperator::local_apply_inner_face_residual,
      &HeatCutOperator::local_apply_boundary_face_residual,
      this,
      residual,
      temperature_old,
      true,
      dealii::MatrixFree<dim, number, dealii::VectorizedArray<number>>::DataAccessOnFaces::
        gradients,
      dealii::MatrixFree<dim, number, dealii::VectorizedArray<number>>::DataAccessOnFaces::
        gradients);
  }



  /*****************************************************************************
   * functions for integral evaluations
   * **************************************************************************/

  // tangent operator domain integral
  template <typename Evaluation>
  inline void
  do_domain_integral_tangent(
    Evaluation                             &evaluator,
    const typename Evaluation::ScalarNumber conductivity,
    const typename Evaluation::ScalarNumber cv, // density * specific_heat_capacity
    const typename DomainEval<Evaluation::dimension,
                              typename Evaluation::ScalarNumber,
                              Evaluation::dimension>::value_type &velocity,
    const typename Evaluation::ScalarNumber                       ost_factor, // delta_t * theta
    const unsigned int                                            q)
  {
    const auto flux_1 = ost_factor * evaluator.get_gradient(q);

    // heat storage
    auto val = cv * evaluator.get_value(q);

    // conductive heat flux
    evaluator.submit_gradient(conductivity * flux_1, q);

    // convective heat flux
    val += cv * dealii::scalar_product(velocity, flux_1);

    evaluator.submit_value(val, q);
  }



  // residual domain integral
  template <typename Evaluation>
  inline void
  do_domain_integral_residual(
    Evaluation                             &evaluator, // T^n+1
    const Evaluation                       &Told_eval,
    const typename Evaluation::ScalarNumber conductivity,
    const typename Evaluation::ScalarNumber cv, // density * specific_heat_capacity
    const typename DomainEval<Evaluation::dimension,
                              typename Evaluation::ScalarNumber,
                              Evaluation::dimension>::value_type &velocity,
    const typename Evaluation::ScalarNumber ost_factor_implicit, // delta_t * theta
    const typename Evaluation::ScalarNumber ost_factor_explicit, // delta_t * (1. - theta)
    const unsigned int                      q)
  {
    const auto flux_1 = ost_factor_implicit * evaluator.get_gradient(q) +
                        ost_factor_explicit * Told_eval.get_gradient(q);

    // heat storage
    auto val = cv * (evaluator.get_value(q) - Told_eval.get_value(q));

    // conductive heat flux
    evaluator.submit_gradient(conductivity * flux_1 * -1., q);

    // convective heat flux
    val += cv * dealii::scalar_product(velocity, flux_1);

    evaluator.submit_value(val * -1., q);
  }



  // tangent operator immersed boundary integral (one-phase case)
  // only if evaporation-induced heat loss is enabled
  template <typename Evaluation>
  inline void
  do_immersed_boundary_integral_tangent(
    Evaluation       &evaluator,
    const Evaluation &T_eval,
    const std::function<typename Evaluation::value_type(typename Evaluation::value_type)>
                                           &compute_qVapor_derivative,
    const typename Evaluation::ScalarNumber ost_factor, // delta_t * theta
    const unsigned int                      q)
  {
    // temperature-dependent evaporation-induced heat flux
    evaluator.submit_value(-ost_factor * compute_qVapor_derivative(T_eval.get_value(q)) *
                             evaluator.get_value(q),
                           q);
  }



  // residual immsersed boundary integral (one-domain case)
  template <typename Evaluation>
  inline void
  do_immersed_boundary_integral_residual(
    Evaluation                        &evaluator, // T^n+1
    [[maybe_unused]] const Evaluation &Told_eval,
    [[maybe_unused]] const std::function<
      typename Evaluation::value_type(typename Evaluation::value_type)> &compute_qVapor,
    [[maybe_unused]] const typename Evaluation::ScalarNumber ost_factor_implicit, // delta_t * theta
    [[maybe_unused]] const typename Evaluation::ScalarNumber
                                           ost_factor_explicit,    // delta_t * (1. - theta)
    const typename Evaluation::value_type &laser_heat_flux_factor, // laser_heat_flux * delta_t
    const bool                             enable_evapor_heat_loss,
    const unsigned int                     q)
  {
    // laser heat flux
    auto val = -laser_heat_flux_factor;

    if (enable_evapor_heat_loss)
      {
        // temperature-dependent evaporation-induced heat flux
        val += -ost_factor_implicit * compute_qVapor(evaluator.get_value(q));

        // temperature-dependent evaporation-induced heat flux
        val += -ost_factor_explicit * compute_qVapor(Told_eval.get_value(q));
      }
    evaluator.submit_value(val * -1., q);
  }



  // tangent operator interface integral (two-domain case)
  template <typename Evaluation>
  inline void
  do_interface_integral_tangent(
    Evaluation                        &evaluator_l,
    Evaluation                        &evaluator_g,
    [[maybe_unused]] const Evaluation &T_eval_l,
    [[maybe_unused]] const Evaluation &T_eval_g,
    [[maybe_unused]] const std::function<
      typename Evaluation::value_type(typename Evaluation::value_type)> &compute_qVapor_derivative,
    const typename Evaluation::ScalarNumber                              conductivity_l,
    const typename Evaluation::ScalarNumber                              conductivity_g,
    const typename Evaluation::ScalarNumber ost_factor,     // delta_t * theta
    const typename Evaluation::ScalarNumber nitsche_factor, // delta_t * gamma_Gamma / h
    const typename Evaluation::ScalarNumber kappa_l,
    const typename Evaluation::ScalarNumber kappa_g,
    const bool                              enable_evapor_heat_loss,
    const unsigned int                      q)
  {
    const auto eval_l   = evaluator_l.get_value(q);
    const auto eval_g   = evaluator_g.get_value(q);
    const auto val_jump = eval_l - eval_g;
    const auto normal_l = evaluator_l.normal_vector(q);
    const auto normal_g = -normal_l;
    const auto grad_l   = evaluator_l.get_gradient(q);
    const auto grad_g   = evaluator_g.get_gradient(q);

    // heat flux over interface - interface integral from divergence theorem
    auto flux_1 = -ost_factor * (kappa_l * conductivity_l * grad_l * normal_l +
                                 kappa_g * conductivity_g * grad_g * normal_g);
    // Nitsche term
    flux_1 += nitsche_factor * val_jump;
    // submit to [v]
    auto val_l = flux_1;
    auto val_g = -flux_1;

    if (enable_evapor_heat_loss)
      {
        // temperature-dependent evaporation-induced heat flux
        val_l += -ost_factor * kappa_g * compute_qVapor_derivative(T_eval_l.get_value(q)) * eval_l;
        val_g += -ost_factor * kappa_l * compute_qVapor_derivative(T_eval_g.get_value(q)) * eval_g;
      }

    // symmetry term
    const auto flux_2 = -ost_factor * val_jump;
    // submit to {k ∂_n v}
    evaluator_l.submit_gradient(flux_2 * conductivity_l * kappa_l * normal_l, q);
    evaluator_g.submit_gradient(flux_2 * conductivity_g * kappa_g * normal_g, q);

    evaluator_l.submit_value(val_l, q);
    evaluator_g.submit_value(val_g, q);
  }



  // residual interface integral (two-domain case)
  template <typename Evaluation>
  inline void
  do_interface_integral_residual(
    Evaluation       &evaluator_l, // T^n+1_l
    Evaluation       &evaluator_g, // T^n+1_g
    const Evaluation &Told_eval_l,
    const Evaluation &Told_eval_g,
    [[maybe_unused]] const std::function<
      typename Evaluation::value_type(typename Evaluation::value_type)> &compute_qVapor,
    const typename Evaluation::ScalarNumber                              conductivity_l,
    const typename Evaluation::ScalarNumber                              conductivity_g,
    const typename Evaluation::ScalarNumber ost_factor_implicit,    // delta_t * theta
    const typename Evaluation::ScalarNumber ost_factor_explicit,    // delta_t * (1. - theta)
    const typename Evaluation::ScalarNumber nitsche_factor,         // delta_t * gamma_Gamma / h
    const typename Evaluation::value_type  &laser_heat_flux_factor, // laser_heat_flux * delta_t
    const typename Evaluation::ScalarNumber kappa_l,
    const typename Evaluation::ScalarNumber kappa_g,
    const bool                              enable_evapor_heat_loss,
    const bool                              do_rhs_symm_term,
    const unsigned int                      q)
  {
    const auto normal_l    = evaluator_l.normal_vector(q);
    const auto normal_g    = -normal_l;
    const auto Tnew_val_l  = evaluator_l.get_value(q);
    const auto Tnew_val_g  = evaluator_g.get_value(q);
    const auto Tnew_jump   = Tnew_val_l - Tnew_val_g;
    const auto Tnew_grad_l = evaluator_l.get_gradient(q);
    const auto Tnew_grad_g = evaluator_g.get_gradient(q);

    // heat flux over interface - interface integral from divergence theorem
    auto flux_1 = -ost_factor_implicit * (kappa_l * conductivity_l * Tnew_grad_l * normal_l +
                                          kappa_g * conductivity_g * Tnew_grad_g * normal_g);
    // Nitsche term
    flux_1 += nitsche_factor * Tnew_jump;

    // symmetry term
    auto flux_2 = -ost_factor_implicit * Tnew_jump;

    // laser heat flux
    auto val_l = -kappa_g * laser_heat_flux_factor;
    auto val_g = -kappa_l * laser_heat_flux_factor;

    if (enable_evapor_heat_loss)
      {
        // temperature-dependent evaporation-induced heat flux
        val_l += -ost_factor_implicit * kappa_g * compute_qVapor(Tnew_val_l);
        val_g += -ost_factor_implicit * kappa_l * compute_qVapor(Tnew_val_g);
      }

    const auto Told_val_l = Told_eval_l.get_value(q);
    const auto Told_val_g = Told_eval_g.get_value(q);
    const auto Told_jump  = Told_val_l - Told_val_g;

    // heat flux over interface - interface integral from divergence theorem
    flux_1 +=
      -ost_factor_explicit * (kappa_l * conductivity_l * Told_eval_l.get_gradient(q) * normal_l +
                              kappa_g * conductivity_g * Told_eval_g.get_gradient(q) * normal_g);

    if (do_rhs_symm_term)
      {
        // symmetry term
        flux_2 += -ost_factor_explicit * Told_jump;
      }

    if (enable_evapor_heat_loss)
      {
        // temperature-dependent evaporation-induced heat flux
        val_l += -ost_factor_explicit * kappa_g * compute_qVapor(Told_val_l);
        val_g += -ost_factor_explicit * kappa_l * compute_qVapor(Told_val_g);
      }

    // submit to [v]
    val_l += flux_1;
    val_g += -flux_1;

    // submit to {k ∂_n v}
    evaluator_l.submit_gradient(flux_2 * conductivity_l * kappa_l * normal_l * -1., q);
    evaluator_g.submit_gradient(flux_2 * conductivity_g * kappa_g * normal_g * -1., q);

    evaluator_l.submit_value(val_l * -1., q);
    evaluator_g.submit_value(val_g * -1., q);
  }



  // ghost-penalty terms for FE_Q elements (p=1)
  template <typename Evaluation>
  inline void
  do_ghost_penalty_terms(Evaluation                             &evaluator_minus,
                         Evaluation                             &evaluator_plus,
                         const typename Evaluation::ScalarNumber conductivity,
                         const typename Evaluation::ScalarNumber ost_factor, // delta_t * theta
                         const typename Evaluation::ScalarNumber h,
                         const typename Evaluation::ScalarNumber gamma_M,
                         const typename Evaluation::ScalarNumber gamma_A,
                         const unsigned int                      q,
                         const typename Evaluation::ScalarNumber factor = 1.0)
  {
    const auto normal_grad_jump =
      evaluator_minus.get_normal_derivative(q) - evaluator_plus.get_normal_derivative(q);

    // stiffness stabilization
    auto gp = ost_factor * gamma_A * conductivity * h / 3. * normal_grad_jump;

    // mass stabilizations
    gp += gamma_M * dealii::Utilities::fixed_power<3>(h) / 3. * normal_grad_jump;

    // submit to [∂_n v]
    evaluator_minus.submit_normal_derivative(gp * factor, q);
    evaluator_plus.submit_normal_derivative(-gp * factor, q);
  }



  /******************************************************************************
   * Local appliers for the consistent tangent operator
   * ***************************************************************************/

  template <int dim, typename number>
  void
  HeatCutOperator<dim, number>::local_apply_domain_tangent(
    const dealii::MatrixFree<dim, number, dealii::VectorizedArray<number>> &matrix_free,
    VectorType                                                             &dst,
    const VectorType                                                       &src,
    const std::pair<unsigned int, unsigned int>                            &cell_range) const
  {
    std::unique_ptr<DomainEval<dim, number, dim>> vel_eval;
    if (velocity)
      vel_eval =
        std::make_unique<DomainEval<dim, number, dim>>(matrix_free, vel_dof_idx, temp_quad_idx);

    const auto cell_category = matrix_free.get_cell_range_category(cell_range);
    if (cell_category == CutUtil::CellCategory::liquid)
      {
        DomainEval<dim, number> eval_l(matrix_free,
                                       temp_dof_idx /*dof_no*/,
                                       temp_quad_idx /*quad_no*/,
                                       0 /*selected component*/,
                                       CutUtil::CellCategory::liquid /*active_fe_index*/);

        for (unsigned int cell_index = cell_range.first; cell_index < cell_range.second;
             ++cell_index)
          {
            eval_l.reinit(cell_index);
            eval_l.gather_evaluate(src,
                                   dealii::EvaluationFlags::values |
                                     dealii::EvaluationFlags::gradients);
            if (vel_eval)
              {
                vel_eval->reinit(cell_index);
                vel_eval->gather_evaluate(*velocity, dealii::EvaluationFlags::values);
              }

            for (const unsigned int q : eval_l.quadrature_point_indices())
              do_domain_integral_tangent(eval_l,
                                         material_data.liquid.thermal_conductivity,
                                         material_data.liquid.density *
                                           material_data.liquid.specific_heat_capacity,
                                         vel_eval ?
                                           vel_eval->get_value(q) :
                                           typename DomainEval<dim, number, dim>::value_type(),
                                         ost_factor_implicit,
                                         q);
            eval_l.integrate_scatter(dealii::EvaluationFlags::values |
                                       dealii::EvaluationFlags::gradients,
                                     dst);
          }
      }
    else if (cell_category == CutUtil::CellCategory::gas and heat_data.cut.two_phase)
      {
        DomainEval<dim, number> eval_g(matrix_free,
                                       temp_dof_idx /*dof_no*/,
                                       temp_quad_idx /*quad_no*/,
                                       1 /*selected component*/,
                                       CutUtil::CellCategory::gas /*active_fe_index*/);

        for (unsigned int cell_index = cell_range.first; cell_index < cell_range.second;
             ++cell_index)
          {
            eval_g.reinit(cell_index);
            eval_g.gather_evaluate(src,
                                   dealii::EvaluationFlags::values |
                                     dealii::EvaluationFlags::gradients);
            if (vel_eval)
              {
                vel_eval->reinit(cell_index);
                vel_eval->gather_evaluate(*velocity, dealii::EvaluationFlags::values);
              }

            for (const unsigned int q : eval_g.quadrature_point_indices())
              do_domain_integral_tangent(eval_g,
                                         material_data.gas.thermal_conductivity,
                                         material_data.gas.density *
                                           material_data.gas.specific_heat_capacity,
                                         vel_eval ?
                                           vel_eval->get_value(q) :
                                           typename DomainEval<dim, number, dim>::value_type(),
                                         ost_factor_implicit,
                                         q);
            eval_g.integrate_scatter(dealii::EvaluationFlags::values |
                                       dealii::EvaluationFlags::gradients,
                                     dst);
          }
      }
    else if (cell_category == CutUtil::CellCategory::intersected and not heat_data.cut.two_phase)
      {
        // use FEEvaluation and FEPointEvaluation in combination for intersected cells
        DomainEval<dim, number> eval_l(matrix_free,
                                       temp_dof_idx /*dof_no*/,
                                       temp_quad_idx /*quad_no*/,
                                       0 /*selected component*/,
                                       CutUtil::CellCategory::intersected /*active_fe_index*/);
        DomainEval<dim, number> T_eval_l(eval_l);
        PointEval<dim, number>  eval_intersected_l(*mapping_info_cells[0], fe_point_temp);
        std::unique_ptr<PointEval<dim, number, dim>> vel_eval_intersected;
        if (vel_eval)
          vel_eval_intersected =
            std::make_unique<PointEval<dim, number, dim>>(*mapping_info_cells[0], fe_point_vel);
        PointEval<dim, number> eval_surface_l(mapping_info_surface, fe_point_temp);
        PointEval<dim, number> T_eval_surface_l(eval_surface_l);

        for (unsigned int cell_index = cell_range.first; cell_index < cell_range.second;
             ++cell_index)
          {
            eval_l.reinit(cell_index);
            eval_l.read_dof_values(src);
            if (evapor_data.evaporative_cooling.enable)
              {
                T_eval_l.reinit(cell_index);
                T_eval_l.read_dof_values(temperature);
              }
            if (vel_eval)
              {
                vel_eval->reinit(cell_index);
                vel_eval->read_dof_values(*velocity);
              }

            for (unsigned int lane = 0;
                 lane < matrix_free.n_active_entries_per_cell_batch(cell_index);
                 ++lane)
              {
                // evaluate for liquid domain integral
                CutUtil::evaluate_intersected_domain<dim, number>(
                  eval_intersected_l,
                  eval_l,
                  dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients,
                  cell_index,
                  lane,
                  n_dofs_per_cell);
                if (evapor_data.evaporative_cooling.enable)
                  {
                    // evaluate for immersed boundary integral
                    CutUtil::evaluate_intersected_domain<dim, number>(eval_surface_l,
                                                                      eval_l,
                                                                      evaluation_flags_surface,
                                                                      cell_index,
                                                                      lane,
                                                                      n_dofs_per_cell);
                    // evaluate T^n+1 for immersed boundary integral
                    CutUtil::evaluate_intersected_domain<dim, number>(T_eval_surface_l,
                                                                      T_eval_l,
                                                                      evaluation_flags_surface,
                                                                      cell_index,
                                                                      lane,
                                                                      n_dofs_per_cell);
                  }
                // evaluate velocity for liquid domain integral
                if (vel_eval)
                  CutUtil::evaluate_intersected_domain<dim, number, dim>(
                    *vel_eval_intersected,
                    *vel_eval,
                    dealii::EvaluationFlags::values,
                    cell_index,
                    lane,
                    n_dofs_per_cell_vel);

                // do inside domain integral
                for (const unsigned int q : eval_intersected_l.quadrature_point_indices())
                  do_domain_integral_tangent(eval_intersected_l,
                                             material_data.liquid.thermal_conductivity,
                                             material_data.liquid.density *
                                               material_data.liquid.specific_heat_capacity,
                                             vel_eval_intersected ?
                                               vel_eval_intersected->get_value(q) :
                                               typename DomainEval<dim, number, dim>::value_type(),
                                             ost_factor_implicit,
                                             q);
                eval_intersected_l.integrate(
                  dealii::StridedArrayView<number, n_lanes>(&eval_l.begin_dof_values()[0][lane],
                                                            n_dofs_per_cell),
                  dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);

                if (evapor_data.evaporative_cooling.enable)
                  {
                    // do immersed boundary integral
                    for (const unsigned int q : eval_surface_l.quadrature_point_indices())
                      do_immersed_boundary_integral_tangent(
                        eval_surface_l,
                        T_eval_surface_l,
                        [&](const dealii::VectorizedArray<number> &T) {
                          return internal::compute_qVapor_derivative(T);
                        },
                        ost_factor_implicit,
                        q);

                    eval_surface_l.integrate(dealii::StridedArrayView<number, n_lanes>(
                                                               &eval_l.begin_dof_values()[0][lane],
                                                               n_dofs_per_cell),
                                                       evaluation_flags_surface, true
                                /*specify flag 'true' for summing the integrated values into the solution values*/);
                  }
              }
            eval_l.distribute_local_to_global(dst);
          }
      }
    else if (cell_category == CutUtil::CellCategory::intersected and heat_data.cut.two_phase)
      {
        // use FEEvaluation and FEPointEvaluation in combination for intersected cells
        DomainEval<dim, number> eval_l(matrix_free,
                                       temp_dof_idx /*dof_no*/,
                                       temp_quad_idx /*quad_no*/,
                                       0 /*selected component*/,
                                       CutUtil::CellCategory::intersected /*active_fe_index*/);
        DomainEval<dim, number> T_eval_l(eval_l);
        PointEval<dim, number>  eval_intersected_l(*mapping_info_cells[0], fe_point_temp);
        std::unique_ptr<PointEval<dim, number, dim>> vel_eval_intersected_l;
        if (vel_eval)
          vel_eval_intersected_l =
            std::make_unique<PointEval<dim, number, dim>>(*mapping_info_cells[0], fe_point_vel);
        PointEval<dim, number>  eval_surface_l(mapping_info_surface, fe_point_temp);
        PointEval<dim, number>  T_eval_surface_l(eval_surface_l);
        DomainEval<dim, number> eval_g(matrix_free,
                                       temp_dof_idx /*dof_no*/,
                                       temp_quad_idx /*quad_no*/,
                                       1 /*selected component*/,
                                       CutUtil::CellCategory::intersected /*active_fe_index*/);
        DomainEval<dim, number> T_eval_g(eval_g);
        PointEval<dim, number>  eval_intersected_g(*mapping_info_cells[1], fe_point_temp);
        std::unique_ptr<PointEval<dim, number, dim>> vel_eval_intersected_g;
        if (vel_eval)
          vel_eval_intersected_g =
            std::make_unique<PointEval<dim, number, dim>>(*mapping_info_cells[1], fe_point_vel);
        PointEval<dim, number> eval_surface_g(eval_surface_l);
        PointEval<dim, number> T_eval_surface_g(eval_surface_l);

        for (unsigned int cell_index = cell_range.first; cell_index < cell_range.second;
             ++cell_index)
          {
            eval_l.reinit(cell_index);
            eval_g.reinit(cell_index);
            eval_l.read_dof_values(src);
            eval_g.read_dof_values(src);
            if (evapor_data.evaporative_cooling.enable)
              {
                T_eval_l.reinit(cell_index);
                T_eval_g.reinit(cell_index);
                T_eval_l.read_dof_values(temperature);
                T_eval_g.read_dof_values(temperature);
              }
            if (vel_eval)
              {
                vel_eval->reinit(cell_index);
                vel_eval->read_dof_values(*velocity);
              }

            for (unsigned int lane = 0;
                 lane < matrix_free.n_active_entries_per_cell_batch(cell_index);
                 ++lane)
              {
                // evaluate for liquid domain integral
                CutUtil::evaluate_intersected_domain<dim, number>(
                  eval_intersected_l,
                  eval_l,
                  dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients,
                  cell_index,
                  lane,
                  n_dofs_per_cell);
                // evaluate liquid side for interface integral
                CutUtil::evaluate_intersected_domain<dim, number>(eval_surface_l,
                                                                  eval_l,
                                                                  evaluation_flags_surface,
                                                                  cell_index,
                                                                  lane,
                                                                  n_dofs_per_cell);
                // evaluate for gas domain integral
                CutUtil::evaluate_intersected_domain<dim, number>(
                  eval_intersected_g,
                  eval_g,
                  dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients,
                  cell_index,
                  lane,
                  n_dofs_per_cell);
                // evaluate gas side for interface integral
                CutUtil::evaluate_intersected_domain<dim, number>(eval_surface_g,
                                                                  eval_g,
                                                                  evaluation_flags_surface,
                                                                  cell_index,
                                                                  lane,
                                                                  n_dofs_per_cell);
                if (evapor_data.evaporative_cooling.enable)
                  {
                    // evaluate T^n+1_l for interface integral
                    CutUtil::evaluate_intersected_domain<dim, number>(
                      T_eval_surface_l,
                      T_eval_l,
                      dealii::EvaluationFlags::values,
                      cell_index,
                      lane,
                      n_dofs_per_cell);
                    // evaluate T^n+1_g for interface integral
                    CutUtil::evaluate_intersected_domain<dim, number>(
                      T_eval_surface_g,
                      T_eval_g,
                      dealii::EvaluationFlags::values,
                      cell_index,
                      lane,
                      n_dofs_per_cell);
                  }
                if (vel_eval)
                  {
                    // evaluate velocity for liquid domain integral
                    CutUtil::evaluate_intersected_domain<dim, number, dim>(
                      *vel_eval_intersected_l,
                      *vel_eval,
                      dealii::EvaluationFlags::values,
                      cell_index,
                      lane,
                      n_dofs_per_cell_vel);
                    // evaluate velocity for gas domain integral
                    CutUtil::evaluate_intersected_domain<dim, number, dim>(
                      *vel_eval_intersected_g,
                      *vel_eval,
                      dealii::EvaluationFlags::values,
                      cell_index,
                      lane,
                      n_dofs_per_cell_vel);
                  }

                // do inside domain integral
                for (const unsigned int q : eval_intersected_l.quadrature_point_indices())
                  do_domain_integral_tangent(eval_intersected_l,
                                             material_data.liquid.thermal_conductivity,
                                             material_data.liquid.density *
                                               material_data.liquid.specific_heat_capacity,
                                             vel_eval_intersected_l ?
                                               vel_eval_intersected_l->get_value(q) :
                                               typename DomainEval<dim, number, dim>::value_type(),
                                             ost_factor_implicit,
                                             q);
                eval_intersected_l.integrate(
                  dealii::StridedArrayView<number, n_lanes>(&eval_l.begin_dof_values()[0][lane],
                                                            n_dofs_per_cell),
                  dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);

                // do outside domain integral
                for (const unsigned int q : eval_intersected_g.quadrature_point_indices())
                  do_domain_integral_tangent(eval_intersected_g,
                                             material_data.gas.thermal_conductivity,
                                             material_data.gas.density *
                                               material_data.gas.specific_heat_capacity,
                                             vel_eval_intersected_g ?
                                               vel_eval_intersected_g->get_value(q) :
                                               typename DomainEval<dim, number, dim>::value_type(),
                                             ost_factor_implicit,
                                             q);
                eval_intersected_g.integrate(
                  dealii::StridedArrayView<number, n_lanes>(&eval_g.begin_dof_values()[0][lane],
                                                            n_dofs_per_cell),
                  dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);

                // do immersed interface integral
                for (const unsigned int q : eval_surface_l.quadrature_point_indices())
                  do_interface_integral_tangent(
                    eval_surface_l,
                    eval_surface_g,
                    T_eval_surface_l,
                    T_eval_surface_g,
                    [&](const dealii::VectorizedArray<number> &T) {
                      return internal::compute_qVapor_derivative(T);
                    },
                    material_data.liquid.thermal_conductivity,
                    material_data.gas.thermal_conductivity,
                    ost_factor_implicit,
                    nitsche_factor,
                    kappa_l,
                    kappa_g,
                    evapor_data.evaporative_cooling.enable,
                    q);

                eval_surface_l.integrate(
                            dealii::StridedArrayView<number, n_lanes>(&eval_l.begin_dof_values()[0][lane],
                                                              n_dofs_per_cell),
                            evaluation_flags_surface,
                            true
                            /*specify flag 'true' for summing the integrated values into the solution values*/);

                eval_surface_g.integrate(
                            dealii::StridedArrayView<number, n_lanes>(&eval_g.begin_dof_values()[0][lane],
                                                              n_dofs_per_cell),
                            evaluation_flags_surface,
                            true
                            /*specify flag 'true' for summing the integrated values into the solution values*/);
              }
            eval_l.distribute_local_to_global(dst);
            eval_g.distribute_local_to_global(dst);
          }
      }
  }



  template <int dim, typename number>
  void
  HeatCutOperator<dim, number>::local_apply_inner_face_tangent(
    const dealii::MatrixFree<dim, number, dealii::VectorizedArray<number>> &matrix_free,
    VectorType                                                             &dst,
    const VectorType                                                       &src,
    const std::pair<unsigned int, unsigned int>                            &face_range) const
  {
    const dealii::EvaluationFlags::EvaluationFlags evaluation_flags =
      dealii::EvaluationFlags::gradients;

    const auto              face_category = matrix_free.get_face_range_category(face_range);
    const CutUtil::FaceType face_type     = CutUtil::get_face_type(face_category);
    if (face_type == CutUtil::FaceType::intersected_face or
        face_type == CutUtil::FaceType::mixed_face_liquid)
      {
        FaceEval<dim, number> eval_minus_l(matrix_free,
                                           true /*is_interior_face*/,
                                           temp_dof_idx /*dof_no*/,
                                           temp_quad_idx /*quad_no*/,
                                           0 /*selected component*/,
                                           CutUtil::CellCategory::liquid /*active_fe_index*/);
        FaceEval<dim, number> eval_plus_l(matrix_free,
                                          false /*is_interior_face*/,
                                          temp_dof_idx /*dof_no*/,
                                          temp_quad_idx /*quad_no*/,
                                          0 /*selected component*/,
                                          CutUtil::CellCategory::liquid /*active_fe_index*/);

        for (unsigned int face_batch = face_range.first; face_batch < face_range.second;
             face_batch++)
          {
            eval_minus_l.reinit(face_batch);
            eval_plus_l.reinit(face_batch);
            eval_minus_l.gather_evaluate(src, evaluation_flags);
            eval_plus_l.gather_evaluate(src, evaluation_flags);

            for (const unsigned int q : eval_minus_l.quadrature_point_indices())
              do_ghost_penalty_terms(eval_minus_l,
                                     eval_plus_l,
                                     material_data.liquid.thermal_conductivity,
                                     ost_factor_implicit,
                                     cell_side_length,
                                     heat_data.cut.ghost_penalty.gamma_M,
                                     heat_data.cut.ghost_penalty.gamma_A,
                                     q);

            eval_minus_l.integrate_scatter(evaluation_flags, dst);
            eval_plus_l.integrate_scatter(evaluation_flags, dst);
          }
      }
    if ((face_type == CutUtil::FaceType::intersected_face or
         face_type == CutUtil::FaceType::mixed_face_gas) and
        heat_data.cut.two_phase)
      {
        FaceEval<dim, number> eval_minus_g(matrix_free,
                                           true /*is_interior_face*/,
                                           temp_dof_idx /*dof_no*/,
                                           temp_quad_idx /*quad_no*/,
                                           1 /*selected component*/,
                                           CutUtil::CellCategory::gas /*active_fe_index*/);

        FaceEval<dim, number> eval_plus_g(matrix_free,
                                          false /*is_interior_face*/,
                                          temp_dof_idx /*dof_no*/,
                                          temp_quad_idx /*quad_no*/,
                                          1 /*selected component*/,
                                          CutUtil::CellCategory::gas /*active_fe_index*/);

        for (unsigned int face_batch = face_range.first; face_batch < face_range.second;
             face_batch++)
          {
            eval_minus_g.reinit(face_batch);
            eval_plus_g.reinit(face_batch);
            eval_minus_g.gather_evaluate(src, evaluation_flags);
            eval_plus_g.gather_evaluate(src, evaluation_flags);

            for (const unsigned int q : eval_minus_g.quadrature_point_indices())
              do_ghost_penalty_terms(eval_minus_g,
                                     eval_plus_g,
                                     material_data.gas.thermal_conductivity,
                                     ost_factor_implicit,
                                     cell_side_length,
                                     heat_data.cut.ghost_penalty.gamma_M,
                                     heat_data.cut.ghost_penalty.gamma_A,
                                     q);

            eval_minus_g.integrate_scatter(evaluation_flags, dst);
            eval_plus_g.integrate_scatter(evaluation_flags, dst);
          }
      }
  }



  template <int dim, typename number>
  void
  HeatCutOperator<dim, number>::local_apply_boundary_face_tangent(
    const dealii::MatrixFree<dim, number, dealii::VectorizedArray<number>> &matrix_free,
    VectorType                                                             &dst,
    const VectorType                                                       &src,
    const std::pair<unsigned int, unsigned int>                            &face_range) const
  {
    (void)matrix_free;
    (void)dst;
    (void)src;
    (void)face_range;
    // TODO temperature-dependent Neumann BC
  }



  /******************************************************************************
   * Local appliers for the residual
   * ***************************************************************************/

  template <int dim, typename number>
  void
  HeatCutOperator<dim, number>::local_apply_domain_residual(
    const dealii::MatrixFree<dim, number, dealii::VectorizedArray<number>> &matrix_free,
    VectorType                                                             &residual,
    const VectorType                                                       &temperature_old,
    const std::pair<unsigned int, unsigned int>                            &cell_range) const
  {
    std::unique_ptr<DomainEval<dim, number, dim>> vel_eval;
    if (velocity)
      vel_eval =
        std::make_unique<DomainEval<dim, number, dim>>(matrix_free, vel_dof_idx, temp_quad_idx);

    const auto cell_category = matrix_free.get_cell_range_category(cell_range);
    if (cell_category == CutUtil::CellCategory::liquid)
      {
        DomainEval<dim, number> Tnew_eval_l(matrix_free,
                                            temp_dof_idx /*dof_no*/,
                                            temp_quad_idx /*quad_no*/,
                                            0 /*selected component*/,
                                            CutUtil::CellCategory::liquid /*active_fe_index*/);
        DomainEval<dim, number> Told_eval_l(Tnew_eval_l);

        for (unsigned int cell_index = cell_range.first; cell_index < cell_range.second;
             ++cell_index)
          {
            Tnew_eval_l.reinit(cell_index);
            Tnew_eval_l.read_dof_values_plain(temperature);
            Tnew_eval_l.evaluate(dealii::EvaluationFlags::values |
                                 dealii::EvaluationFlags::gradients);
            Told_eval_l.reinit(cell_index);
            Told_eval_l.read_dof_values_plain(temperature_old);
            Told_eval_l.evaluate(dealii::EvaluationFlags::values |
                                 dealii::EvaluationFlags::gradients);
            if (vel_eval)
              {
                vel_eval->reinit(cell_index);
                vel_eval->read_dof_values_plain(*velocity);
                vel_eval->evaluate(dealii::EvaluationFlags::values);
              }

            for (const unsigned int q : Tnew_eval_l.quadrature_point_indices())
              do_domain_integral_residual(Tnew_eval_l,
                                          Told_eval_l,
                                          material_data.liquid.thermal_conductivity,
                                          material_data.liquid.density *
                                            material_data.liquid.specific_heat_capacity,
                                          vel_eval ?
                                            vel_eval->get_value(q) :
                                            typename DomainEval<dim, number, dim>::value_type(),
                                          ost_factor_implicit,
                                          ost_factor_explicit,
                                          q);
            Tnew_eval_l.integrate_scatter(dealii::EvaluationFlags::values |
                                            dealii::EvaluationFlags::gradients,
                                          residual);
          }
      }
    else if (cell_category == CutUtil::CellCategory::gas and heat_data.cut.two_phase)
      {
        DomainEval<dim, number> Tnew_eval_g(matrix_free,
                                            temp_dof_idx /*dof_no*/,
                                            temp_quad_idx /*quad_no*/,
                                            1 /*selected component*/,
                                            CutUtil::CellCategory::gas /*active_fe_index*/);
        DomainEval<dim, number> Told_eval_g(Tnew_eval_g);

        for (unsigned int cell_index = cell_range.first; cell_index < cell_range.second;
             ++cell_index)
          {
            Tnew_eval_g.reinit(cell_index);
            Tnew_eval_g.read_dof_values_plain(temperature);
            Tnew_eval_g.evaluate(dealii::EvaluationFlags::values |
                                 dealii::EvaluationFlags::gradients);
            Told_eval_g.reinit(cell_index);
            Told_eval_g.read_dof_values_plain(temperature_old);
            Told_eval_g.evaluate(dealii::EvaluationFlags::values |
                                 dealii::EvaluationFlags::gradients);
            if (vel_eval)
              {
                vel_eval->reinit(cell_index);
                vel_eval->read_dof_values_plain(*velocity);
                vel_eval->evaluate(dealii::EvaluationFlags::values);
              }

            for (const unsigned int q : Tnew_eval_g.quadrature_point_indices())
              do_domain_integral_residual(Tnew_eval_g,
                                          Told_eval_g,
                                          material_data.gas.thermal_conductivity,
                                          material_data.gas.density *
                                            material_data.gas.specific_heat_capacity,
                                          vel_eval ?
                                            vel_eval->get_value(q) :
                                            typename DomainEval<dim, number, dim>::value_type(),
                                          ost_factor_implicit,
                                          ost_factor_explicit,
                                          q);
            Tnew_eval_g.integrate_scatter(dealii::EvaluationFlags::values |
                                            dealii::EvaluationFlags::gradients,
                                          residual);
          }
      }
    else if (cell_category == CutUtil::CellCategory::intersected and not heat_data.cut.two_phase)
      {
        // use FEEvaluation and FEPointEval<dim, number>uation in combination for intersected cells
        DomainEval<dim, number> Tnew_eval_l(matrix_free,
                                            temp_dof_idx /*dof_no*/,
                                            temp_quad_idx /*quad_no*/,
                                            0 /*selected component*/,
                                            CutUtil::CellCategory::intersected /*active_fe_index*/);
        DomainEval<dim, number> Told_eval_l(Tnew_eval_l);
        PointEval<dim, number>  Tnew_eval_intersected_l(*mapping_info_cells[0], fe_point_temp);
        PointEval<dim, number>  Told_eval_intersected_l(Tnew_eval_intersected_l);
        std::unique_ptr<PointEval<dim, number, dim>> vel_eval_intersected;
        if (vel_eval)
          vel_eval_intersected =
            std::make_unique<PointEval<dim, number, dim>>(*mapping_info_cells[0], fe_point_vel);
        PointEval<dim, number> Tnew_eval_surface_l(mapping_info_surface, fe_point_temp);
        PointEval<dim, number> Told_eval_surface_l(Tnew_eval_surface_l);

        for (unsigned int cell_index = cell_range.first; cell_index < cell_range.second;
             ++cell_index)
          {
            Tnew_eval_l.reinit(cell_index);
            Tnew_eval_l.read_dof_values_plain(temperature);
            Told_eval_l.reinit(cell_index);
            Told_eval_l.read_dof_values_plain(temperature_old);
            if (vel_eval)
              {
                vel_eval->reinit(cell_index);
                vel_eval->read_dof_values_plain(*velocity);
              }

            for (unsigned int lane = 0;
                 lane < matrix_free.n_active_entries_per_cell_batch(cell_index);
                 ++lane)
              {
                // evaluate T^n+1 for liquid domain integral
                CutUtil::evaluate_intersected_domain<dim, number>(
                  Tnew_eval_intersected_l,
                  Tnew_eval_l,
                  dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients,
                  cell_index,
                  lane,
                  n_dofs_per_cell);
                // evaluate T^n+1 for immersed boundary integral
                CutUtil::evaluate_intersected_domain<dim, number>(
                  Tnew_eval_surface_l,
                  Tnew_eval_l,
                  evapor_data.evaporative_cooling.enable ? evaluation_flags_surface :
                                                           dealii::EvaluationFlags::nothing,
                  cell_index,
                  lane,
                  n_dofs_per_cell);
                // evaluate T^n for liquid domain integral
                CutUtil::evaluate_intersected_domain<dim, number>(
                  Told_eval_intersected_l,
                  Told_eval_l,
                  dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients,
                  cell_index,
                  lane,
                  n_dofs_per_cell);
                if (evapor_data.evaporative_cooling.enable)
                  // evaluate T^n for immersed boundary integral
                  CutUtil::evaluate_intersected_domain<dim, number>(Told_eval_surface_l,
                                                                    Told_eval_l,
                                                                    evaluation_flags_surface,
                                                                    cell_index,
                                                                    lane,
                                                                    n_dofs_per_cell);
                // evaluate velocity for liquid domain integral
                if (vel_eval)
                  CutUtil::evaluate_intersected_domain<dim, number, dim>(
                    *vel_eval_intersected,
                    *vel_eval,
                    dealii::EvaluationFlags::values,
                    cell_index,
                    lane,
                    n_dofs_per_cell_vel);


                // do liquid domain integral
                for (const unsigned int q : Tnew_eval_intersected_l.quadrature_point_indices())
                  do_domain_integral_residual(Tnew_eval_intersected_l,
                                              Told_eval_intersected_l,
                                              material_data.liquid.thermal_conductivity,
                                              material_data.liquid.density *
                                                material_data.liquid.specific_heat_capacity,
                                              vel_eval_intersected ?
                                                vel_eval_intersected->get_value(q) :
                                                typename DomainEval<dim, number, dim>::value_type(),
                                              ost_factor_implicit,
                                              ost_factor_explicit,
                                              q);

                Tnew_eval_intersected_l.integrate(dealii::StridedArrayView<number, n_lanes>(
                                                    &Tnew_eval_l.begin_dof_values()[0][lane],
                                                    n_dofs_per_cell),
                                                  dealii::EvaluationFlags::values |
                                                    dealii::EvaluationFlags::gradients);

                // do immersed boundary integral
                for (const unsigned int q : Tnew_eval_surface_l.quadrature_point_indices())
                  do_immersed_boundary_integral_residual(
                    Tnew_eval_surface_l,
                    Told_eval_surface_l,
                    [&](const dealii::VectorizedArray<number> &T) {
                      return internal::compute_qVapor(T);
                    },
                    ost_factor_implicit,
                    ost_factor_explicit,
                    internal::compute_laser_heat_source(
                      laser_intensity_profile.get(), laser_direction, Tnew_eval_surface_l, q) *
                      this->time_increment,
                    evapor_data.evaporative_cooling.enable,
                    q);

                Tnew_eval_surface_l.integrate(
                            dealii::StridedArrayView<number, n_lanes>(
                                    &Tnew_eval_l.begin_dof_values()[0][lane], n_dofs_per_cell),
                            evaluation_flags_surface,
                            true
                            /*specify flag 'true' for summing the integrated values into the solution values*/);
              }

            Tnew_eval_l.distribute_local_to_global(residual);
          }
      }
    else if (cell_category == CutUtil::CellCategory::intersected and heat_data.cut.two_phase)
      {
        // use FEEvaluation and FEPointEval<dim, number>uation in combination for intersected cells
        DomainEval<dim, number> Tnew_eval_l(matrix_free,
                                            temp_dof_idx /*dof_no*/,
                                            temp_quad_idx /*quad_no*/,
                                            0 /*selected component*/,
                                            CutUtil::CellCategory::intersected /*active_fe_index*/);
        DomainEval<dim, number> Told_eval_l(Tnew_eval_l);
        PointEval<dim, number>  Tnew_eval_intersected_l(*mapping_info_cells[0], fe_point_temp);
        PointEval<dim, number>  Told_eval_intersected_l(Tnew_eval_intersected_l);
        std::unique_ptr<PointEval<dim, number, dim>> vel_eval_intersected_l;
        if (vel_eval)
          vel_eval_intersected_l =
            std::make_unique<PointEval<dim, number, dim>>(*mapping_info_cells[0], fe_point_vel);
        PointEval<dim, number>  Tnew_eval_surface_l(mapping_info_surface, fe_point_temp);
        PointEval<dim, number>  Told_eval_surface_l(Tnew_eval_surface_l);
        DomainEval<dim, number> Tnew_eval_g(matrix_free,
                                            temp_dof_idx /*dof_no*/,
                                            temp_quad_idx /*quad_no*/,
                                            1 /*selected component*/,
                                            CutUtil::CellCategory::intersected /*active_fe_index*/);
        DomainEval<dim, number> Told_eval_g(Tnew_eval_g);
        PointEval<dim, number>  Tnew_eval_intersected_g(*mapping_info_cells[1], fe_point_temp);
        PointEval<dim, number>  Told_eval_intersected_g(Tnew_eval_intersected_g);
        std::unique_ptr<PointEval<dim, number, dim>> vel_eval_intersected_g;
        if (vel_eval)
          vel_eval_intersected_g =
            std::make_unique<PointEval<dim, number, dim>>(*mapping_info_cells[1], fe_point_vel);
        PointEval<dim, number> Tnew_eval_surface_g(Tnew_eval_surface_l);
        PointEval<dim, number> Told_eval_surface_g(Tnew_eval_surface_l);

        for (unsigned int cell_index = cell_range.first; cell_index < cell_range.second;
             ++cell_index)
          {
            const auto reinit_and_read = [&](DomainEval<dim, number> &evaluator,
                                             const VectorType        &dof_vector) {
              evaluator.reinit(cell_index);
              evaluator.read_dof_values_plain(dof_vector);
            };

            reinit_and_read(Tnew_eval_l, temperature);
            reinit_and_read(Tnew_eval_g, temperature);
            reinit_and_read(Told_eval_l, temperature_old);
            reinit_and_read(Told_eval_g, temperature_old);
            if (vel_eval)
              {
                vel_eval->reinit(cell_index);
                vel_eval->read_dof_values_plain(*velocity);
              }

            for (unsigned int lane = 0;
                 lane < matrix_free.n_active_entries_per_cell_batch(cell_index);
                 ++lane)
              {
                // evaluate T^n+1_l for liquid domain integral
                CutUtil::evaluate_intersected_domain<dim, number>(
                  Tnew_eval_intersected_l,
                  Tnew_eval_l,
                  dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients,
                  cell_index,
                  lane,
                  n_dofs_per_cell);
                // evaluate T^n+1_l for interface integral
                CutUtil::evaluate_intersected_domain<dim, number>(Tnew_eval_surface_l,
                                                                  Tnew_eval_l,
                                                                  evaluation_flags_surface,
                                                                  cell_index,
                                                                  lane,
                                                                  n_dofs_per_cell);
                // evaluate T^n+1_g for gas domain integral
                CutUtil::evaluate_intersected_domain<dim, number>(
                  Tnew_eval_intersected_g,
                  Tnew_eval_g,
                  dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients,
                  cell_index,
                  lane,
                  n_dofs_per_cell);
                // evaluate T^n+1_g for interface integral
                CutUtil::evaluate_intersected_domain<dim, number>(Tnew_eval_surface_g,
                                                                  Tnew_eval_g,
                                                                  evaluation_flags_surface,
                                                                  cell_index,
                                                                  lane,
                                                                  n_dofs_per_cell);
                // evaluate T^n_l for liquid domain integral
                CutUtil::evaluate_intersected_domain<dim, number>(
                  Told_eval_intersected_l,
                  Told_eval_l,
                  dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients,
                  cell_index,
                  lane,
                  n_dofs_per_cell);
                // evaluate T^n_l for interface integral
                CutUtil::evaluate_intersected_domain<dim, number>(Told_eval_surface_l,
                                                                  Told_eval_l,
                                                                  evaluation_flags_surface,
                                                                  cell_index,
                                                                  lane,
                                                                  n_dofs_per_cell);
                // evaluate T^n_g for gas domain integral
                CutUtil::evaluate_intersected_domain<dim, number>(
                  Told_eval_intersected_g,
                  Told_eval_g,
                  dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients,
                  cell_index,
                  lane,
                  n_dofs_per_cell);
                // evaluate T^n_g for interface integral
                CutUtil::evaluate_intersected_domain<dim, number>(Told_eval_surface_g,
                                                                  Told_eval_g,
                                                                  evaluation_flags_surface,
                                                                  cell_index,
                                                                  lane,
                                                                  n_dofs_per_cell);
                if (vel_eval)
                  {
                    // evaluate velocity for liquid domain integral
                    CutUtil::evaluate_intersected_domain<dim, number, dim>(
                      *vel_eval_intersected_l,
                      *vel_eval,
                      dealii::EvaluationFlags::values,
                      cell_index,
                      lane,
                      n_dofs_per_cell_vel);
                    // evaluate velocity for gas domain integral
                    CutUtil::evaluate_intersected_domain<dim, number, dim>(
                      *vel_eval_intersected_g,
                      *vel_eval,
                      dealii::EvaluationFlags::values,
                      cell_index,
                      lane,
                      n_dofs_per_cell_vel);
                  }

                // do liquid domain integral
                for (const unsigned int q : Tnew_eval_intersected_l.quadrature_point_indices())
                  do_domain_integral_residual(Tnew_eval_intersected_l,
                                              Told_eval_intersected_l,
                                              material_data.liquid.thermal_conductivity,
                                              material_data.liquid.density *
                                                material_data.liquid.specific_heat_capacity,
                                              vel_eval_intersected_l ?
                                                vel_eval_intersected_l->get_value(q) :
                                                typename DomainEval<dim, number, dim>::value_type(),
                                              ost_factor_implicit,
                                              ost_factor_explicit,
                                              q);

                Tnew_eval_intersected_l.integrate(dealii::StridedArrayView<number, n_lanes>(
                                                    &Tnew_eval_l.begin_dof_values()[0][lane],
                                                    n_dofs_per_cell),
                                                  dealii::EvaluationFlags::values |
                                                    dealii::EvaluationFlags::gradients);

                // do gas domain integral
                for (const unsigned int q : Tnew_eval_intersected_g.quadrature_point_indices())
                  do_domain_integral_residual(Tnew_eval_intersected_g,
                                              Told_eval_intersected_g,
                                              material_data.gas.thermal_conductivity,
                                              material_data.gas.density *
                                                material_data.gas.specific_heat_capacity,
                                              vel_eval_intersected_g ?
                                                vel_eval_intersected_g->get_value(q) :
                                                typename DomainEval<dim, number, dim>::value_type(),
                                              ost_factor_implicit,
                                              ost_factor_explicit,
                                              q);
                Tnew_eval_intersected_g.integrate(dealii::StridedArrayView<number, n_lanes>(
                                                    &Tnew_eval_g.begin_dof_values()[0][lane],
                                                    n_dofs_per_cell),
                                                  dealii::EvaluationFlags::values |
                                                    dealii::EvaluationFlags::gradients);

                // do interface integral
                for (const unsigned int q : Tnew_eval_surface_l.quadrature_point_indices())
                  do_interface_integral_residual(
                    Tnew_eval_surface_l,
                    Tnew_eval_surface_g,
                    Told_eval_surface_l,
                    Told_eval_surface_g,
                    [&](const dealii::VectorizedArray<number> &T) {
                      return internal::compute_qVapor(T);
                    },
                    material_data.liquid.thermal_conductivity,
                    material_data.gas.thermal_conductivity,
                    ost_factor_implicit,
                    ost_factor_explicit,
                    nitsche_factor,
                    internal::compute_laser_heat_source(
                      laser_intensity_profile.get(), laser_direction, Tnew_eval_surface_l, q) *
                      this->time_increment,
                    kappa_l,
                    kappa_g,
                    evapor_data.evaporative_cooling.enable,
                    heat_data.cut.do_explicit_symmetry_term,
                    q);

                Tnew_eval_surface_l.integrate(
                            dealii::StridedArrayView<number, n_lanes>(
                                    &Tnew_eval_l.begin_dof_values()[0][lane], n_dofs_per_cell),
                            evaluation_flags_surface,
                            true
                            /*specify flag 'true' for summing the integrated values into the solution values*/);

                Tnew_eval_surface_g.integrate(
                            dealii::StridedArrayView<number, n_lanes>(
                                    &Tnew_eval_g.begin_dof_values()[0][lane], n_dofs_per_cell),
                            evaluation_flags_surface,
                            true
                            /*specify flag 'true' for summing the integrated values into the solution values*/);
              }

            Tnew_eval_l.distribute_local_to_global(residual);
            Tnew_eval_g.distribute_local_to_global(residual);
          }
      }
  }



  template <int dim, typename number>
  void
  HeatCutOperator<dim, number>::local_apply_inner_face_residual(
    const dealii::MatrixFree<dim, number, dealii::VectorizedArray<number>> &matrix_free,
    VectorType                                                             &residual,
    [[maybe_unused]] const VectorType                                      &temperature_old,
    const std::pair<unsigned int, unsigned int>                            &face_range) const
  {
    const dealii::EvaluationFlags::EvaluationFlags evaluation_flags =
      dealii::EvaluationFlags::gradients;

    const auto              face_category = matrix_free.get_face_range_category(face_range);
    const CutUtil::FaceType face_type     = CutUtil::get_face_type(face_category);
    if (face_type == CutUtil::FaceType::intersected_face or
        face_type == CutUtil::FaceType::mixed_face_liquid)
      {
        FaceEval<dim, number> Tnew_eval_minus_l(matrix_free,
                                                true /*is_interior_face*/,
                                                temp_dof_idx /*dof_no*/,
                                                temp_quad_idx /*quad_no*/,
                                                0 /*selected component*/,
                                                CutUtil::CellCategory::liquid /*active_fe_index*/);
        FaceEval<dim, number> Tnew_eval_plus_l(matrix_free,
                                               false /*is_interior_face*/,
                                               temp_dof_idx /*dof_no*/,
                                               temp_quad_idx /*quad_no*/,
                                               0 /*selected component*/,
                                               CutUtil::CellCategory::liquid /*active_fe_index*/);

        for (unsigned int face_batch = face_range.first; face_batch < face_range.second;
             face_batch++)
          {
            Tnew_eval_minus_l.reinit(face_batch);
            Tnew_eval_plus_l.reinit(face_batch);
            Tnew_eval_minus_l.gather_evaluate(temperature, evaluation_flags);
            Tnew_eval_plus_l.gather_evaluate(temperature, evaluation_flags);

            for (const unsigned int q : Tnew_eval_minus_l.quadrature_point_indices())
              do_ghost_penalty_terms(Tnew_eval_minus_l,
                                     Tnew_eval_plus_l,
                                     material_data.liquid.thermal_conductivity,
                                     ost_factor_implicit,
                                     cell_side_length,
                                     heat_data.cut.ghost_penalty.gamma_M,
                                     heat_data.cut.ghost_penalty.gamma_A,
                                     q,
                                     -1.0);

            Tnew_eval_minus_l.integrate_scatter(evaluation_flags, residual);
            Tnew_eval_plus_l.integrate_scatter(evaluation_flags, residual);
          }
      }
    if ((face_type == CutUtil::FaceType::intersected_face or
         face_type == CutUtil::FaceType::mixed_face_gas) and
        heat_data.cut.two_phase)
      {
        for (unsigned int face_batch = face_range.first; face_batch < face_range.second;
             face_batch++)
          {
            FaceEval<dim, number> eval_minus_g(matrix_free,
                                               true /*is_interior_face*/,
                                               temp_dof_idx /*dof_no*/,
                                               temp_quad_idx /*quad_no*/,
                                               1 /*selected component*/,
                                               CutUtil::CellCategory::gas /*active_fe_index*/);

            FaceEval<dim, number> eval_plus_g(matrix_free,
                                              false /*is_interior_face*/,
                                              temp_dof_idx /*dof_no*/,
                                              temp_quad_idx /*quad_no*/,
                                              1 /*selected component*/,
                                              CutUtil::CellCategory::gas /*active_fe_index*/);

            eval_minus_g.reinit(face_batch);
            eval_plus_g.reinit(face_batch);
            eval_minus_g.gather_evaluate(temperature, evaluation_flags);
            eval_plus_g.gather_evaluate(temperature, evaluation_flags);

            for (const unsigned int q : eval_minus_g.quadrature_point_indices())
              do_ghost_penalty_terms(eval_minus_g,
                                     eval_plus_g,
                                     material_data.gas.thermal_conductivity,
                                     ost_factor_implicit,
                                     cell_side_length,
                                     heat_data.cut.ghost_penalty.gamma_M,
                                     heat_data.cut.ghost_penalty.gamma_A,
                                     q,
                                     -1.0);

            eval_minus_g.integrate_scatter(evaluation_flags, residual);
            eval_plus_g.integrate_scatter(evaluation_flags, residual);
          }
      }
  }



  template <int dim, typename number>
  void
  HeatCutOperator<dim, number>::local_apply_boundary_face_residual(
    const dealii::MatrixFree<dim, number, dealii::VectorizedArray<number>> &matrix_free,
    VectorType                                                             &residual,
    const VectorType                                                       &temperature_old,
    const std::pair<unsigned int, unsigned int>                            &face_range) const
  {
    (void)matrix_free;
    (void)residual;
    (void)temperature_old;
    (void)face_range;
    // TODO non-zero Neumann BC
  }



  template <int dim, typename number>
  void
  HeatCutOperator<dim, number>::compute_inverse_diagonal_from_matrixfree(VectorType &diagonal) const
  {
    // TODO: isn't here the DoF index missing?
    scratch_data.initialize_dof_vector(diagonal, temp_dof_idx);

    dealii::TrilinosWrappers::SparseMatrix dummy;
    internal_compute_diagonal_or_system_matrix(diagonal, dummy, true);

    // invert
    const double linfty_norm = std::max(1.0, diagonal.linfty_norm());
    for (auto &i : diagonal)
      i = (std::abs(i) > 1.0e-14 * linfty_norm) ? (1.0 / i) : 1.0;
  }



  template <int dim, typename number>
  void
  HeatCutOperator<dim, number>::compute_system_matrix_from_matrixfree(
    dealii::TrilinosWrappers::SparseMatrix &system_matrix) const
  {
    system_matrix = 0.0;

    VectorType dummy;
    internal_compute_diagonal_or_system_matrix(dummy, system_matrix, false);
  }



  template <int dim, typename number>
  void
  HeatCutOperator<dim, number>::internal_compute_diagonal_or_system_matrix(
    [[maybe_unused]] VectorType                             &diagonal,
    [[maybe_unused]] dealii::TrilinosWrappers::SparseMatrix &system_matrix,
    const bool                                               do_diagonal) const
  {
    const auto &matrix_free = scratch_data.get_matrix_free();

    // use FEEvaluation and FEPointEvaluation in combination for intersected cells
    DomainEval<dim, number> eval_l(matrix_free,
                                   temp_dof_idx /*dof_no*/,
                                   0 /*quad_no*/,
                                   0 /*selected component*/,
                                   CutUtil::CellCategory::intersected /*active_fe_index*/);
    DomainEval<dim, number> T_eval_l(eval_l);
    PointEval<dim, number>  eval_intersected_l(*mapping_info_cells[0], fe_point_temp);
    PointEval<dim, number>  eval_surface_l(mapping_info_surface, fe_point_temp);
    PointEval<dim, number>  T_eval_surface_l(eval_surface_l);
    // the following short hand if else is a workaround to avoid a segfault for the one phase case
    // in which mapping_info_cells only contains a single phase.
    DomainEval<dim, number> eval_g(matrix_free,
                                   temp_dof_idx /*dof_no*/,
                                   0 /*quad_no*/,
                                   heat_data.cut.two_phase ? 1 : 0 /*selected component*/,
                                   CutUtil::CellCategory::intersected /*active_fe_index*/);
    DomainEval<dim, number> T_eval_g(eval_g);
    PointEval<dim, number>  eval_intersected_g(*mapping_info_cells[heat_data.cut.two_phase ? 1 : 0],
                                              fe_point_temp);
    PointEval<dim, number>  eval_surface_g(eval_surface_l);
    PointEval<dim, number>  T_eval_surface_g(eval_surface_l);

    unsigned int old_cell_index = numbers::invalid_unsigned_int;

    dealii::MatrixFreeTools::internal::
      ComputeMatrixScratchData<dim, dealii::VectorizedArray<number>, false /*is_face_*/>
        data_cell;
    dealii::MatrixFreeTools::internal::
      ComputeMatrixScratchData<dim, dealii::VectorizedArray<number>, true /*is_face_*/>
        data_face;

    if (heat_data.cut.two_phase)
      {
        data_cell.dof_numbers               = {temp_dof_idx, temp_dof_idx};
        data_cell.quad_numbers              = {0, 0};
        data_cell.n_components              = {1, 1};
        data_cell.first_selected_components = {0, 1};
        data_cell.batch_type                = {0, 0}; // 0 for cell

        data_face.dof_numbers  = {temp_dof_idx, temp_dof_idx, temp_dof_idx, temp_dof_idx};
        data_face.quad_numbers = {0, 0, 0, 0};
        data_face.n_components = {1, 1, 1, 1};
        data_face.first_selected_components = {0, 0, 1, 1};
        data_face.batch_type = {1, 2, 1, 2}; // 1 for interior face, 2 for exterior face
      }
    else
      {
        data_cell.dof_numbers               = {temp_dof_idx};
        data_cell.quad_numbers              = {0};
        data_cell.n_components              = {1};
        data_cell.first_selected_components = {0};
        data_cell.batch_type                = {0}; // 0 for cell

        data_face.dof_numbers               = {temp_dof_idx, temp_dof_idx};
        data_face.quad_numbers              = {0, 0};
        data_face.n_components              = {1, 1};
        data_face.first_selected_components = {0, 0};
        data_face.batch_type                = {1, 2}; // 1 for interior face, 2 for exterior face
      }



    data_cell.op_create = [&](const std::pair<unsigned int, unsigned int> &cell_range) {
      std::vector<
        std::unique_ptr<dealii::FEEvaluationData<dim, dealii::VectorizedArray<number>, false>>>
        eval_data;

      const auto emplace_eval = [&](const unsigned int index) {
        eval_data.emplace_back(
          std::make_unique<DomainEval<dim, number>>(matrix_free,
                                                    cell_range,
                                                    data_cell.dof_numbers[index],
                                                    data_cell.quad_numbers[index],
                                                    data_cell.first_selected_components[index]));
      };

      const auto cell_category = matrix_free.get_cell_range_category(cell_range);

      if (cell_category == CutUtil::CellCategory::liquid or
          cell_category == CutUtil::CellCategory::intersected)
        {
          emplace_eval(0);
        }

      if ((cell_category == CutUtil::CellCategory::gas or
           cell_category == CutUtil::CellCategory::intersected) and
          heat_data.cut.two_phase)
        {
          emplace_eval(1);
        }

      return eval_data;
    };



    data_cell.op_reinit = [&](auto &evaluators, const unsigned cell_index) {
      for (unsigned int i = 0; i < evaluators.size(); ++i)
        static_cast<DomainEval<dim, number> &>(*evaluators[i]).reinit(cell_index);
    };



    data_cell.op_compute =
      [&](auto &evaluators) {
        auto &eval_1 = static_cast<DomainEval<dim, number> &>(*evaluators[0]);

        const unsigned int cell_index    = eval_1.get_current_cell_index();
        const unsigned int cell_category = eval_1.get_active_fe_index();

        std::unique_ptr<DomainEval<dim, number, dim>> vel_eval;
        if (velocity)
          {
            vel_eval = std::make_unique<DomainEval<dim, number, dim>>(matrix_free,
                                                                      vel_dof_idx,
                                                                      temp_quad_idx);
            vel_eval->reinit(cell_index);
            vel_eval->read_dof_values(*velocity);
          }

        if (cell_category == CutUtil::CellCategory::liquid)
          {
            eval_1.evaluate(dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);
            if (vel_eval)
              vel_eval->evaluate(dealii::EvaluationFlags::values);

            for (const unsigned int q : eval_1.quadrature_point_indices())
              do_domain_integral_tangent(eval_1,
                                         material_data.liquid.thermal_conductivity,
                                         material_data.liquid.density *
                                           material_data.liquid.specific_heat_capacity,
                                         vel_eval ?
                                           vel_eval->get_value(q) :
                                           typename DomainEval<dim, number, dim>::value_type(),
                                         ost_factor_implicit,
                                         q);
            eval_1.integrate(dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);
          }
        else if (cell_category == CutUtil::CellCategory::gas and heat_data.cut.two_phase)
          {
            eval_1.evaluate(dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);
            if (vel_eval)
              vel_eval->evaluate(dealii::EvaluationFlags::values);

            for (const unsigned int q : eval_1.quadrature_point_indices())
              do_domain_integral_tangent(eval_1,
                                         material_data.gas.thermal_conductivity,
                                         material_data.gas.density *
                                           material_data.gas.specific_heat_capacity,
                                         vel_eval ?
                                           vel_eval->get_value(q) :
                                           typename DomainEval<dim, number, dim>::value_type(),
                                         ost_factor_implicit,
                                         q);
            eval_1.integrate(dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);
          }
        else if (cell_category == CutUtil::CellCategory::intersected and
                 not heat_data.cut.two_phase)
          {
            std::unique_ptr<PointEval<dim, number, dim>> vel_eval_intersected;
            if (vel_eval)
              vel_eval_intersected =
                std::make_unique<PointEval<dim, number, dim>>(*mapping_info_cells[0], fe_point_vel);

            if (evapor_data.evaporative_cooling.enable and cell_index != old_cell_index)
              {
                T_eval_l.reinit(cell_index);
                T_eval_l.read_dof_values(temperature);
              }

            for (unsigned int lane = 0;
                 lane < matrix_free.n_active_entries_per_cell_batch(cell_index);
                 ++lane)
              {
                // evaluate for inside domain integral
                CutUtil::evaluate_intersected_domain<dim, number>(
                  eval_intersected_l,
                  eval_1,
                  dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients,
                  cell_index,
                  lane,
                  n_dofs_per_cell);
                if (evapor_data.evaporative_cooling.enable)
                  {
                    // evaluate for inside surface integral
                    CutUtil::evaluate_intersected_domain<dim, number>(eval_surface_l,
                                                                      eval_1,
                                                                      evaluation_flags_surface,
                                                                      cell_index,
                                                                      lane,
                                                                      n_dofs_per_cell);
                    // evaluate T^n+1 for inside surface integral
                    CutUtil::evaluate_intersected_domain<dim, number>(T_eval_surface_l,
                                                                      T_eval_l,
                                                                      evaluation_flags_surface,
                                                                      cell_index,
                                                                      lane,
                                                                      n_dofs_per_cell);
                  }
                // evaluate velocity for liquid domain integral
                if (vel_eval)
                  CutUtil::evaluate_intersected_domain<dim, number, dim>(
                    *vel_eval_intersected,
                    *vel_eval,
                    dealii::EvaluationFlags::values,
                    cell_index,
                    lane,
                    n_dofs_per_cell_vel);

                // do inside domain integral
                for (const unsigned int q : eval_intersected_l.quadrature_point_indices())
                  do_domain_integral_tangent(eval_intersected_l,
                                             material_data.liquid.thermal_conductivity,
                                             material_data.liquid.density *
                                               material_data.liquid.specific_heat_capacity,
                                             vel_eval_intersected ?
                                               vel_eval_intersected->get_value(q) :
                                               typename DomainEval<dim, number, dim>::value_type(),
                                             ost_factor_implicit,
                                             q);
                eval_intersected_l.integrate(
                  dealii::StridedArrayView<number, n_lanes>(&eval_1.begin_dof_values()[0][lane],
                                                            n_dofs_per_cell),
                  dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);

                if (evapor_data.evaporative_cooling.enable)
                  {
                    // do immersed boundary integral
                    for (const unsigned int q : eval_surface_l.quadrature_point_indices())
                      do_immersed_boundary_integral_tangent(
                        eval_surface_l,
                        T_eval_surface_l,
                        [&](const dealii::VectorizedArray<number> &T) {
                          return internal::compute_qVapor_derivative(T);
                        },
                        ost_factor_implicit,
                        q);

                    eval_surface_l.integrate(dealii::StridedArrayView<number, n_lanes>(
                                                     &eval_1.begin_dof_values()[0][lane],
                                                     n_dofs_per_cell),
                                                   evaluation_flags_surface, true
                                                   /*specify flag 'true' for summing the integrated values into the solution values*/);
                  }
              }
          }
        else if (cell_category == CutUtil::CellCategory::intersected and heat_data.cut.two_phase)
          {
            auto &eval_2 = static_cast<DomainEval<dim, number> &>(*evaluators[1]);

            std::unique_ptr<PointEval<dim, number, dim>> vel_eval_intersected_l;
            std::unique_ptr<PointEval<dim, number, dim>> vel_eval_intersected_g;
            if (vel_eval)
              {
                vel_eval_intersected_l =
                  std::make_unique<PointEval<dim, number, dim>>(*mapping_info_cells[0],
                                                                fe_point_vel);
                vel_eval_intersected_g =
                  std::make_unique<PointEval<dim, number, dim>>(*mapping_info_cells[1],
                                                                fe_point_vel);
              }

            if (evapor_data.evaporative_cooling.enable and cell_index != old_cell_index)
              {
                T_eval_l.reinit(cell_index);
                T_eval_g.reinit(cell_index);
                T_eval_l.read_dof_values(temperature);
                T_eval_g.read_dof_values(temperature);
              }

            for (unsigned int lane = 0;
                 lane < matrix_free.n_active_entries_per_cell_batch(cell_index);
                 ++lane)
              {
                // evaluate for inside domain integral
                CutUtil::evaluate_intersected_domain<dim, number>(
                  eval_intersected_l,
                  eval_1,
                  dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients,
                  cell_index,
                  lane,
                  n_dofs_per_cell);
                // evaluate for inside surface integral
                CutUtil::evaluate_intersected_domain<dim, number>(eval_surface_l,
                                                                  eval_1,
                                                                  evaluation_flags_surface,
                                                                  cell_index,
                                                                  lane,
                                                                  n_dofs_per_cell);
                // evaluate for outside domain integral
                CutUtil::evaluate_intersected_domain<dim, number>(
                  eval_intersected_g,
                  eval_2,
                  dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients,
                  cell_index,
                  lane,
                  n_dofs_per_cell);
                // evaluate for outside surface integral
                CutUtil::evaluate_intersected_domain<dim, number>(eval_surface_g,
                                                                  eval_2,
                                                                  evaluation_flags_surface,
                                                                  cell_index,
                                                                  lane,
                                                                  n_dofs_per_cell);
                if (evapor_data.evaporative_cooling.enable)
                  {
                    // evaluate T^n+1_l for inside surface integral
                    CutUtil::evaluate_intersected_domain<dim, number>(T_eval_surface_l,
                                                                      T_eval_l,
                                                                      EvaluationFlags::values,
                                                                      cell_index,
                                                                      lane,
                                                                      n_dofs_per_cell);
                    // evaluate T^n+1_g for inside surface integral
                    CutUtil::evaluate_intersected_domain<dim, number>(T_eval_surface_g,
                                                                      T_eval_g,
                                                                      EvaluationFlags::values,
                                                                      cell_index,
                                                                      lane,
                                                                      n_dofs_per_cell);
                  }
                if (vel_eval)
                  {
                    // evaluate velocity for liquid domain integral
                    CutUtil::evaluate_intersected_domain<dim, number, dim>(
                      *vel_eval_intersected_l,
                      *vel_eval,
                      dealii::EvaluationFlags::values,
                      cell_index,
                      lane,
                      n_dofs_per_cell_vel);
                    // evaluate velocity for gas domain integral
                    CutUtil::evaluate_intersected_domain<dim, number, dim>(
                      *vel_eval_intersected_g,
                      *vel_eval,
                      dealii::EvaluationFlags::values,
                      cell_index,
                      lane,
                      n_dofs_per_cell_vel);
                  }

                // do inside domain integral
                for (const unsigned int q : eval_intersected_l.quadrature_point_indices())
                  do_domain_integral_tangent(eval_intersected_l,
                                             material_data.liquid.thermal_conductivity,
                                             material_data.liquid.density *
                                               material_data.liquid.specific_heat_capacity,
                                             vel_eval_intersected_l ?
                                               vel_eval_intersected_l->get_value(q) :
                                               typename DomainEval<dim, number, dim>::value_type(),
                                             ost_factor_implicit,
                                             q);
                eval_intersected_l.integrate(
                  dealii::StridedArrayView<number, n_lanes>(&eval_1.begin_dof_values()[0][lane],
                                                            n_dofs_per_cell),
                  dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);

                // do outside domain integral
                for (const unsigned int q : eval_intersected_g.quadrature_point_indices())
                  do_domain_integral_tangent(eval_intersected_g,
                                             material_data.gas.thermal_conductivity,
                                             material_data.gas.density *
                                               material_data.gas.specific_heat_capacity,
                                             vel_eval_intersected_g ?
                                               vel_eval_intersected_g->get_value(q) :
                                               typename DomainEval<dim, number, dim>::value_type(),
                                             ost_factor_implicit,
                                             q);
                eval_intersected_g.integrate(
                  dealii::StridedArrayView<number, n_lanes>(&eval_2.begin_dof_values()[0][lane],
                                                            n_dofs_per_cell),
                  dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);

                // do immersed interface integral
                for (const unsigned int q : eval_surface_l.quadrature_point_indices())
                  do_interface_integral_tangent(
                    eval_surface_l,
                    eval_surface_g,
                    T_eval_surface_l,
                    T_eval_surface_g,
                    [&](const dealii::VectorizedArray<number> &T) {
                      return internal::compute_qVapor_derivative(T);
                    },
                    material_data.liquid.thermal_conductivity,
                    material_data.gas.thermal_conductivity,
                    ost_factor_implicit,
                    nitsche_factor,
                    kappa_l,
                    kappa_g,
                    evapor_data.evaporative_cooling.enable,
                    q);

                eval_surface_l.integrate(
                    dealii::StridedArrayView<number, n_lanes>(&eval_1.begin_dof_values()[0][lane],
                                                      n_dofs_per_cell),
                    evaluation_flags_surface,
                    true
                    /*specify flag 'true' for summing the integrated values into the solution values*/);

                eval_surface_g.integrate(
                    dealii::StridedArrayView<number, n_lanes>(&eval_2.begin_dof_values()[0][lane],
                                                      n_dofs_per_cell),
                    evaluation_flags_surface,
                    true
                    /*specify flag 'true' for summing the integrated values into the solution values*/);
              }
          }
        old_cell_index = cell_index;
      };



    data_face.op_create = [&](const std::pair<unsigned int, unsigned int> &face_range) {
      std::vector<std::unique_ptr<FEEvaluationData<dim, VectorizedArray<number>, true>>> eval_data;

      const auto emplace_face_eval = [&](const unsigned int index) {
        bool       is_interior_face;
        const auto batch_type = data_face.batch_type[index];
        if (batch_type == 1)
          is_interior_face = true;
        else if (batch_type == 2)
          is_interior_face = false;
        else
          AssertThrow(
            false,
            dealii::ExcMessage(
              "The face batch type must either be 1 (interior face) or 2 (exterior face)!"));
        eval_data.emplace_back(
          std::make_unique<FaceEval<dim, number>>(matrix_free,
                                                  face_range,
                                                  is_interior_face,
                                                  data_face.dof_numbers[index],
                                                  data_face.quad_numbers[index],
                                                  data_face.first_selected_components[index]));
      };

      const auto              face_category = matrix_free.get_face_range_category(face_range);
      const CutUtil::FaceType face_type     = CutUtil::get_face_type(face_category);

      if (face_type == CutUtil::FaceType ::intersected_face or
          face_type == CutUtil::FaceType ::mixed_face_liquid)
        {
          emplace_face_eval(0);
          emplace_face_eval(1);
        }
      if ((face_type == CutUtil::FaceType ::intersected_face or
           face_type == CutUtil::FaceType ::mixed_face_gas) and
          heat_data.cut.two_phase)
        {
          emplace_face_eval(2);
          emplace_face_eval(3);
        }
      return eval_data;
    };



    data_face.op_reinit = [](auto &evaluators, const unsigned face_index) {
      for (unsigned int i = 0; i < evaluators.size(); ++i)
        static_cast<FaceEval<dim, number> &>(*evaluators[i]).reinit(face_index);
    };



    data_face.op_compute = [&](auto &evaluators) {
      auto &eval_minus_l = static_cast<FaceEval<dim, number> &>(*evaluators[0]);
      auto &eval_plus_l  = static_cast<FaceEval<dim, number> &>(*evaluators[1]);

      const dealii::EvaluationFlags::EvaluationFlags evaluation_flags =
        dealii::EvaluationFlags::gradients;

      const unsigned int      face_index    = eval_minus_l.get_cell_or_face_batch_id();
      const auto              face_category = matrix_free.get_face_category(face_index);
      const CutUtil::FaceType face_type     = CutUtil::get_face_type(face_category);

      if (face_type == CutUtil::FaceType ::intersected_face or
          face_type == CutUtil::FaceType ::mixed_face_liquid)
        {
          eval_minus_l.evaluate(evaluation_flags);
          eval_plus_l.evaluate(evaluation_flags);

          for (const unsigned int q : eval_minus_l.quadrature_point_indices())
            do_ghost_penalty_terms(eval_minus_l,
                                   eval_plus_l,
                                   material_data.liquid.thermal_conductivity,
                                   ost_factor_implicit,
                                   cell_side_length,
                                   heat_data.cut.ghost_penalty.gamma_M,
                                   heat_data.cut.ghost_penalty.gamma_A,
                                   q);

          eval_minus_l.integrate(evaluation_flags);
          eval_plus_l.integrate(evaluation_flags);
        }
      if ((face_type == CutUtil::FaceType ::intersected_face or
           face_type == CutUtil::FaceType ::mixed_face_gas) and
          heat_data.cut.two_phase)
        {
          const int eval_idx = evaluators.size() == 4 ? 2 : 0;

          auto &eval_minus_g = static_cast<FaceEval<dim, number> &>(*evaluators[eval_idx]);
          auto &eval_plus_g  = static_cast<FaceEval<dim, number> &>(*evaluators[eval_idx + 1]);

          eval_minus_g.evaluate(evaluation_flags);
          eval_plus_g.evaluate(evaluation_flags);

          for (const unsigned int q : eval_minus_g.quadrature_point_indices())
            do_ghost_penalty_terms(eval_minus_g,
                                   eval_plus_g,
                                   material_data.gas.thermal_conductivity,
                                   ost_factor_implicit,
                                   cell_side_length,
                                   heat_data.cut.ghost_penalty.gamma_M,
                                   heat_data.cut.ghost_penalty.gamma_A,
                                   q);

          eval_minus_g.integrate(evaluation_flags);
          eval_plus_g.integrate(evaluation_flags);
        }
    };

    if (do_diagonal)
      {
        std::vector<VectorType *> dummy(1);
        dummy[0] = &diagonal;
        dealii::MatrixFreeTools::internal::
          compute_diagonal<dim, number, dealii::VectorizedArray<number>>(
            matrix_free, data_cell, data_face, {} /*data_boundary*/, diagonal, dummy);
      }
    else // compute matrix
      {
        dealii::MatrixFreeTools::internal::
          compute_matrix<dim, number, dealii::VectorizedArray<number>>(matrix_free,
                                                                       scratch_data.get_constraint(
                                                                         temp_dof_idx),
                                                                       data_cell,
                                                                       data_face,
                                                                       {} /*data_boundary*/,
                                                                       system_matrix);
      }
  }



  template <int dim, typename number>
  number
  HeatCutOperator<dim, number>::compute_cut_L2_norm(const VectorType &solution) const
  {
    if (not solution.has_ghost_elements())
      solution.update_ghost_values();

    number error_L2_squared = 0.;

    const auto &matrix_free = scratch_data.get_matrix_free();
    for (unsigned int cell_index = 0; cell_index < matrix_free.n_cell_batches(); ++cell_index)
      {
        const auto cell_category = matrix_free.get_cell_category(cell_index);
        if (cell_category == CutUtil::CellCategory::liquid)
          {
            DomainEval<dim, number> eval_l(matrix_free,
                                           temp_hanging_nodes_dof_idx /*dof_no*/,
                                           0 /*quad_no*/,
                                           0 /*selected component*/,
                                           CutUtil::CellCategory::liquid /*active_fe_index*/);

            eval_l.reinit(cell_index);
            eval_l.gather_evaluate(solution, dealii::EvaluationFlags::values);

            for (const unsigned int q : eval_l.quadrature_point_indices())
              for (unsigned int lane = 0;
                   lane < matrix_free.n_active_entries_per_cell_batch(cell_index);
                   ++lane)
                error_L2_squared += dealii::Utilities::fixed_power<2>(eval_l.get_value(q)[lane]) *
                                    eval_l.JxW(q)[lane];
          }
        else if (cell_category == CutUtil::CellCategory::gas and heat_data.cut.two_phase)
          {
            DomainEval<dim, number> eval_g(matrix_free,
                                           temp_hanging_nodes_dof_idx /*dof_no*/,
                                           0 /*quad_no*/,
                                           1 /*selected component*/,
                                           CutUtil::CellCategory::gas /*active_fe_index*/);

            eval_g.reinit(cell_index);
            eval_g.gather_evaluate(solution, dealii::EvaluationFlags::values);

            for (const unsigned int q : eval_g.quadrature_point_indices())
              for (unsigned int lane = 0;
                   lane < matrix_free.n_active_entries_per_cell_batch(cell_index);
                   ++lane)
                error_L2_squared += dealii::Utilities::fixed_power<2>(eval_g.get_value(q)[lane]) *
                                    eval_g.JxW(q)[lane];
          }
        else if (cell_category == CutUtil::CellCategory::intersected and
                 not heat_data.cut.two_phase)
          {
            // use FEEvaluation and FEPointEvaluation in combination for intersected cells
            DomainEval<dim, number> eval_l(matrix_free,
                                           temp_hanging_nodes_dof_idx /*dof_no*/,
                                           0 /*quad_no*/,
                                           0 /*selected component*/,
                                           CutUtil::CellCategory::liquid /*active_fe_index*/);
            PointEval<dim, number>  eval_intersected_l(*mapping_info_cells[0], fe_point_temp);

            eval_l.reinit(cell_index);
            eval_l.read_dof_values_plain(solution);

            for (unsigned int lane = 0;
                 lane < matrix_free.n_active_entries_per_cell_batch(cell_index);
                 ++lane)
              {
                CutUtil::evaluate_intersected_domain<dim, number>(eval_intersected_l,
                                                                  eval_l,
                                                                  dealii::EvaluationFlags::values,
                                                                  cell_index,
                                                                  lane,
                                                                  n_dofs_per_cell);

                for (const unsigned int q : eval_intersected_l.quadrature_point_indices())
                  for (unsigned int v = 0; v < n_lanes; ++v)
                    error_L2_squared +=
                      dealii::Utilities::fixed_power<2>(eval_intersected_l.get_value(q)[v]) *
                      eval_intersected_l.JxW(q)[v];
              }
          }
        else if (cell_category == CutUtil::CellCategory::intersected and heat_data.cut.two_phase)
          {
            // use FEEvaluation and FEPointEvaluation in combination for intersected cells
            DomainEval<dim, number> eval_l(matrix_free,
                                           temp_hanging_nodes_dof_idx /*dof_no*/,
                                           0 /*quad_no*/,
                                           0 /*selected component*/,
                                           CutUtil::CellCategory::liquid /*active_fe_index*/);
            PointEval<dim, number>  eval_intersected_l(*mapping_info_cells[0], fe_point_temp);
            DomainEval<dim, number> eval_g(matrix_free,
                                           temp_hanging_nodes_dof_idx /*dof_no*/,
                                           0 /*quad_no*/,
                                           1 /*selected component*/,
                                           CutUtil::CellCategory::gas /*active_fe_index*/);
            PointEval<dim, number>  eval_intersected_g(*mapping_info_cells[1], fe_point_temp);

            eval_l.reinit(cell_index);
            eval_g.reinit(cell_index);
            eval_l.read_dof_values_plain(solution);
            eval_g.read_dof_values_plain(solution);

            for (unsigned int lane = 0;
                 lane < matrix_free.n_active_entries_per_cell_batch(cell_index);
                 ++lane)
              {
                CutUtil::evaluate_intersected_domain<dim, number>(eval_intersected_l,
                                                                  eval_l,
                                                                  EvaluationFlags::values,
                                                                  cell_index,
                                                                  lane,
                                                                  n_dofs_per_cell);
                CutUtil::evaluate_intersected_domain<dim, number>(eval_intersected_g,
                                                                  eval_g,
                                                                  EvaluationFlags::values,
                                                                  cell_index,
                                                                  lane,
                                                                  n_dofs_per_cell);

                for (const unsigned int q : eval_intersected_l.quadrature_point_indices())
                  for (unsigned int v = 0; v < n_lanes; ++v)
                    error_L2_squared +=
                      dealii::Utilities::fixed_power<2>(eval_intersected_l.get_value(q)[v]) *
                      eval_intersected_l.JxW(q)[v];

                for (const unsigned int q : eval_intersected_g.quadrature_point_indices())
                  for (unsigned int v = 0; v < n_lanes; ++v)
                    error_L2_squared +=
                      dealii::Utilities::fixed_power<2>(eval_intersected_g.get_value(q)[v]) *
                      eval_intersected_g.JxW(q)[v];
              }
          }
      }

    return std::sqrt(
      dealii::Utilities::MPI::sum(error_L2_squared, solution.get_mpi_communicator()));
  }



  template class HeatCutOperator<1, double>;
  template class HeatCutOperator<2, double>;
  template class HeatCutOperator<3, double>;
} // namespace MeltPoolDG::Heat
