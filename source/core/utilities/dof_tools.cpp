#include <meltpooldg/utilities/dof_tools.hpp>
//
#include <deal.II/base/exceptions.h>
#include <deal.II/base/point.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_q_dg0.h>
#include <deal.II/fe/fe_q_iso_q1.h>

#include <deal.II/matrix_free/evaluation_flags.h>

#include <vector>


namespace MeltPoolDG::DoFTools
{
  template <int dim, typename number>
  dealii::FullMatrix<number>
  create_dof_interpolation_matrix(const dealii::DoFHandler<dim> &dof_handler_1, // e.g. pressure
                                  const dealii::DoFHandler<dim> &dof_handler_2, // e.g. level set
                                  const bool                     do_sort_lexicographically)
  {
    dealii::FullMatrix<number> dof_interpolation_matrix(dof_handler_1.get_fe().n_dofs_per_cell(),
                                                        dof_handler_2.get_fe().n_dofs_per_cell());

    const dealii::FE_Q_iso_Q1<dim> *fe_2 =
      dynamic_cast<const dealii::FE_Q_iso_Q1<dim> *>(&dof_handler_2.get_fe());

    AssertThrow(
      fe_2, dealii::ExcMessage("dof_handler_2 must contain finite elements of type FE_Q_iso_Q1."));

    const std::vector<unsigned int> lexicographic_ls = fe_2->get_poly_space_numbering_inverse();


    //@todo: get rid of base_element
    if (const dealii::FE_Q<dim> *fe_1 =
          dynamic_cast<const dealii::FE_Q<dim> *>(&dof_handler_1.get_fe().base_element(0)))
      {
        const std::vector<unsigned int> lexicographic_p = fe_1->get_poly_space_numbering_inverse();
        for (unsigned int j = 0; j < fe_1->dofs_per_cell; ++j)
          {
            const dealii::Point<dim> p =
              fe_1->get_unit_support_points()[do_sort_lexicographically ? lexicographic_p[j] : j];
            for (unsigned int i = 0; i < fe_2->dofs_per_cell; ++i)
              dof_interpolation_matrix(j, i) = dof_handler_2.get_fe().shape_value(
                do_sort_lexicographically ? lexicographic_ls[i] : i, p);
          }
      }
    else if (const dealii::FE_Q_DG0<dim> *fe_1_dq0 =
               dynamic_cast<const dealii::FE_Q_DG0<dim> *>(&dof_handler_1.get_fe().base_element(0)))
      {
        const std::vector<unsigned int> lexicographic_p =
          fe_1_dq0->get_poly_space_numbering_inverse();

        // Loop over all support points except the one for the discontinuous
        // shape function in the middle of the cell (dofs_per_cell - 1).
        for (unsigned int j = 0; j < fe_1_dq0->dofs_per_cell - 1; ++j)
          {
            const dealii::Point<dim> p =
              fe_1_dq0
                ->get_unit_support_points()[do_sort_lexicographically ? lexicographic_p[j] : j];
            for (unsigned int i = 0; i < fe_2->dofs_per_cell; ++i)
              dof_interpolation_matrix(j, i) = dof_handler_2.get_fe().shape_value(
                do_sort_lexicographically ? lexicographic_ls[i] : i, p);
          }
      }
    else
      AssertThrow(false,
                  dealii::ExcMessage("The operation for the requested pair of DoFHandler "
                                     "is not supported. Types must be: dof_handler_1 = FE_Q_iso_Q1;"
                                     " dof_handler_2 = FE_Q || FE_Q_DG0."));

    return dof_interpolation_matrix;
  }

  template <int dim, typename number>
  void
  compute_gradient_at_interpolated_dof_values(
    FECellIntegrator<dim, 1, number> &values,
    FECellIntegrator<dim, 1, number> &interpolated_values,
    const dealii::FullMatrix<number> &interpolation_matrix)
  {
    // Evaluate the field Φ at the support points of its space j
    values.evaluate(dealii::EvaluationFlags::values);

    // Loop over the support points of the to be interpolated space i
    for (unsigned int i = 0; i < interpolated_values.dofs_per_cell; ++i)
      {
        dealii::VectorizedArray<number> interpolated_value = 0;

        // Interpolate the field Φ from the support points of the space j
        // of the original field to the one of the interpolated field i,
        // using the interpolation matrix P
        // _
        // Φ   = P   · Φ
        //  i     ij    j
        for (unsigned int j = 0; j < values.dofs_per_cell; ++j)
          interpolated_value += interpolation_matrix(i, j) * values.get_dof_value(j);

        // Store the interpolated values at the support points of the pressure space
        interpolated_values.submit_dof_value(interpolated_value, i);
      }

    // Evaluate the gradient from the interpolated field
    //                       _
    //                     ∇ Φ
    //
    interpolated_values.evaluate(dealii::EvaluationFlags::gradients);
  }


  template dealii::FullMatrix<double>
  create_dof_interpolation_matrix(const dealii::DoFHandler<1> &,
                                  const dealii::DoFHandler<1> &,
                                  const bool);
  template dealii::FullMatrix<double>
  create_dof_interpolation_matrix(const dealii::DoFHandler<2> &,
                                  const dealii::DoFHandler<2> &,
                                  const bool);
  template dealii::FullMatrix<double>
  create_dof_interpolation_matrix(const dealii::DoFHandler<3> &,
                                  const dealii::DoFHandler<3> &,
                                  const bool);

  template void
  compute_gradient_at_interpolated_dof_values(FECellIntegrator<1, 1, double> &,
                                              FECellIntegrator<1, 1, double> &,
                                              const dealii::FullMatrix<double> &);
  template void
  compute_gradient_at_interpolated_dof_values(FECellIntegrator<2, 1, double> &,
                                              FECellIntegrator<2, 1, double> &,
                                              const dealii::FullMatrix<double> &);
  template void
  compute_gradient_at_interpolated_dof_values(FECellIntegrator<3, 1, double> &,
                                              FECellIntegrator<3, 1, double> &,
                                              const dealii::FullMatrix<double> &);
} // namespace MeltPoolDG::DoFTools