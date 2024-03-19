#include <deal.II/base/exceptions.h>
#include <deal.II/base/quadrature_lib.h>

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
  distribute_dofs(const FiniteElementData &fe_data, DoFHandler<dim> &dof_handler)
  {
    if constexpr (n_components == 1)
      {
        switch (fe_data.type)
          {
              case FiniteElementType::FE_Q: {
                dof_handler.distribute_dofs(FE_Q<dim>(fe_data.degree));
                break;
              }
              case FiniteElementType::FE_SimplexP: {
                dof_handler.distribute_dofs(FE_SimplexP<dim>(fe_data.degree));
                break;
              }
              case FiniteElementType::FE_Q_iso_Q1: {
                dof_handler.distribute_dofs(FE_Q_iso_Q1<dim>(fe_data.degree));
                break;
              }
            case FiniteElementType::not_initialized:
              DEAL_II_ASSERT_UNREACHABLE();
            case FiniteElementType::FE_DGQ:
            default:
              DEAL_II_NOT_IMPLEMENTED();
          }
      }
    else
      {
        switch (fe_data.type)
          {
              case FiniteElementType::FE_Q: {
                dof_handler.distribute_dofs(FESystem<dim>(FE_Q<dim>(fe_data.degree), n_components));
                break;
              }
              case FiniteElementType::FE_SimplexP: {
                dof_handler.distribute_dofs(
                  FESystem<dim>(FE_SimplexP<dim>(fe_data.degree), n_components));
                break;
              }
              case FiniteElementType::FE_Q_iso_Q1: {
                dof_handler.distribute_dofs(
                  FESystem<dim>(FE_Q_iso_Q1<dim>(fe_data.degree), n_components));
                break;
              }
            case FiniteElementType::not_initialized:
              DEAL_II_ASSERT_UNREACHABLE();
            case FiniteElementType::FE_DGQ:
            default:
              DEAL_II_NOT_IMPLEMENTED();
          }
      }
  }

  template <int dim>
  std::shared_ptr<Mapping<dim>>
  create_mapping(const FiniteElementData &fe_data)
  {
    switch (fe_data.type)
      {
        case FiniteElementType::FE_Q:
          return std::make_shared<MappingQGeneric<dim>>(fe_data.degree);
        case FiniteElementType::FE_Q_iso_Q1:
          return std::make_shared<MappingQGeneric<dim>>(1);
        case FiniteElementType::FE_SimplexP:
          return std::make_shared<MappingFE<dim>>(FE_SimplexP<dim>(fe_data.degree));
        case FiniteElementType::not_initialized:
          DEAL_II_ASSERT_UNREACHABLE();
        case FiniteElementType::FE_DGQ:
        default:
          DEAL_II_NOT_IMPLEMENTED();
      }
    return nullptr;
  }

  template <int dim>
  Quadrature<dim>
  create_quadrature(const FiniteElementData &fe_data)
  {
    switch (fe_data.type)
      {
        case FiniteElementType::FE_Q:
          return QGauss<dim>(fe_data.get_n_q_points());
        case FiniteElementType::FE_SimplexP:
          return QGaussSimplex<dim>(fe_data.get_n_q_points());
        case FiniteElementType::FE_Q_iso_Q1:
          return QIterated<dim>(QGauss<1>(2), fe_data.degree);
        case FiniteElementType::not_initialized:
          DEAL_II_ASSERT_UNREACHABLE();
        case FiniteElementType::FE_DGQ:
        default:
          DEAL_II_NOT_IMPLEMENTED();
      }
    return Quadrature<dim>();
  }

  template void
  distribute_dofs<1, 1>(const FiniteElementData &, DoFHandler<1> &);
  template void
  distribute_dofs<2, 1>(const FiniteElementData &, DoFHandler<2> &);
  template void
  distribute_dofs<2, 2>(const FiniteElementData &, DoFHandler<2> &);
  template void
  distribute_dofs<3, 1>(const FiniteElementData &, DoFHandler<3> &);
  template void
  distribute_dofs<3, 3>(const FiniteElementData &, DoFHandler<3> &);

  template std::shared_ptr<Mapping<1>>
  create_mapping(const FiniteElementData &);
  template std::shared_ptr<Mapping<2>>
  create_mapping(const FiniteElementData &);
  template std::shared_ptr<Mapping<3>>
  create_mapping(const FiniteElementData &);

  template Quadrature<1>
  create_quadrature(const FiniteElementData &);
  template Quadrature<2>
  create_quadrature(const FiniteElementData &);
  template Quadrature<3>
  create_quadrature(const FiniteElementData &);
} // namespace MeltPoolDG::FiniteElementUtils