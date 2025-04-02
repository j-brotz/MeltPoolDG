#include <deal.II/base/exceptions.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_q_iso_q1.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_q_generic.h>

#include <meltpooldg/utilities/fe_util.hpp>

namespace MeltPoolDG::FiniteElementUtils
{
  template <int dim, unsigned int n_components>
  void
  distribute_dofs(const FiniteElementData &fe_data, dealii::DoFHandler<dim> &dof_handler)
  {
    if constexpr (n_components == 1)
      {
        switch (fe_data.type)
          {
              case FiniteElementType::FE_Q: {
                dof_handler.distribute_dofs(dealii::FE_Q<dim>(fe_data.degree));
                break;
              }
              case FiniteElementType::FE_SimplexP: {
                dof_handler.distribute_dofs(dealii::FE_SimplexP<dim>(fe_data.degree));
                break;
              }
              case FiniteElementType::FE_Q_iso_Q1: {
                dof_handler.distribute_dofs(dealii::FE_Q_iso_Q1<dim>(fe_data.degree));
                break;
              }
              case FiniteElementType::FE_DGQ: {
                dof_handler.distribute_dofs(dealii::FE_DGQ<dim>(fe_data.degree));
                break;
              }
            case FiniteElementType::not_initialized:
              DEAL_II_ASSERT_UNREACHABLE();
            default:
              DEAL_II_NOT_IMPLEMENTED();
          }
      }
    else
      {
        switch (fe_data.type)
          {
              case FiniteElementType::FE_Q: {
                dof_handler.distribute_dofs(
                  dealii::FESystem<dim>(dealii::FE_Q<dim>(fe_data.degree), n_components));
                break;
              }
              case FiniteElementType::FE_SimplexP: {
                dof_handler.distribute_dofs(
                  dealii::FESystem<dim>(dealii::FE_SimplexP<dim>(fe_data.degree), n_components));
                break;
              }
              case FiniteElementType::FE_Q_iso_Q1: {
                dof_handler.distribute_dofs(
                  dealii::FESystem<dim>(dealii::FE_Q_iso_Q1<dim>(fe_data.degree), n_components));
                break;
              }
              case FiniteElementType::FE_DGQ: {
                dof_handler.distribute_dofs(
                  dealii::FESystem<dim>(dealii::FE_DGQ<dim>(fe_data.degree), n_components));
                break;
              }
            case FiniteElementType::not_initialized:
              DEAL_II_ASSERT_UNREACHABLE();
            default:
              DEAL_II_NOT_IMPLEMENTED();
          }
      }
  }

  template <int dim>
  std::shared_ptr<dealii::Mapping<dim>>
  create_mapping(const FiniteElementData &fe_data)
  {
    switch (fe_data.type)
      {
        case FiniteElementType::FE_Q:
          return std::make_shared<dealii::MappingQGeneric<dim>>(fe_data.degree);
        case FiniteElementType::FE_Q_iso_Q1:
          return std::make_shared<dealii::MappingQGeneric<dim>>(1);
        case FiniteElementType::FE_SimplexP:
          return std::make_shared<dealii::MappingFE<dim>>(dealii::FE_SimplexP<dim>(fe_data.degree));
        case FiniteElementType::FE_DGQ:
          return std::make_shared<dealii::MappingQGeneric<dim>>(fe_data.degree);
        case FiniteElementType::not_initialized:
          DEAL_II_ASSERT_UNREACHABLE();
        default:
          DEAL_II_NOT_IMPLEMENTED();
      }
    return nullptr;
  }

  template <int dim>
  dealii::Quadrature<dim>
  create_quadrature(const FiniteElementData &fe_data)
  {
    switch (fe_data.type)
      {
        case FiniteElementType::FE_Q:
          return dealii::QGauss<dim>(fe_data.get_n_q_points());
        case FiniteElementType::FE_SimplexP:
          return dealii::QGaussSimplex<dim>(fe_data.get_n_q_points());
        case FiniteElementType::FE_Q_iso_Q1:
          return dealii::QIterated<dim>(dealii::QGauss<1>(2), fe_data.degree);
        case FiniteElementType::FE_DGQ:
          return dealii::QGauss<dim>(fe_data.get_n_q_points());
        case FiniteElementType::not_initialized:
          DEAL_II_ASSERT_UNREACHABLE();
        default:
          DEAL_II_NOT_IMPLEMENTED();
      }
    return dealii::Quadrature<dim>();
  }

  template void
  distribute_dofs<1, 1>(const FiniteElementData &, dealii::DoFHandler<1> &);
  template void
  distribute_dofs<1, 3>(const FiniteElementData &, dealii::DoFHandler<1> &);
  template void
  distribute_dofs<2, 1>(const FiniteElementData &, dealii::DoFHandler<2> &);
  template void
  distribute_dofs<2, 2>(const FiniteElementData &, dealii::DoFHandler<2> &);
  template void
  distribute_dofs<2, 4>(const FiniteElementData &, dealii::DoFHandler<2> &);
  template void
  distribute_dofs<3, 1>(const FiniteElementData &, dealii::DoFHandler<3> &);
  template void
  distribute_dofs<3, 3>(const FiniteElementData &, dealii::DoFHandler<3> &);
  template void
  distribute_dofs<3, 5>(const FiniteElementData &, dealii::DoFHandler<3> &);

  template std::shared_ptr<dealii::Mapping<1>>
  create_mapping(const FiniteElementData &);
  template std::shared_ptr<dealii::Mapping<2>>
  create_mapping(const FiniteElementData &);
  template std::shared_ptr<dealii::Mapping<3>>
  create_mapping(const FiniteElementData &);

  template dealii::Quadrature<1>
  create_quadrature(const FiniteElementData &);
  template dealii::Quadrature<2>
  create_quadrature(const FiniteElementData &);
  template dealii::Quadrature<3>
  create_quadrature(const FiniteElementData &);
} // namespace MeltPoolDG::FiniteElementUtils