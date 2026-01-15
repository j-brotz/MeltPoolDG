#pragma once

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/tensor.h>

#include <meltpooldg/particles/dem_util.hpp>
#include <meltpooldg/particles/obstacle_field.hpp>
#include <meltpooldg/particles/particle_accessor.hpp>


namespace MeltPoolDG
{
  template <typename number>
  struct SphericalParticleContactData
  {
    /// Coefficient of restitution for damping in particle collisions.
    number restitution_coefficient;

    struct MaterialData
    {
      /// Young's modulus of the particle material.
      number youngs_modulus;

      /// Poisson's ratio of the particle material.
      number poisson_ratio;
    };

    /// Material data for the particles. Those are assumed to be identical for all particles.
    MaterialData particle;

    /**
     * Add the relevant parameters to a parameter handler.
     *
     * @param prm The parameter handler to which the parameters are added.
     */
    void
    add_parameters(dealii::ParameterHandler &prm);
  };


  template <int dim, typename number, typename ObstacleType>
  class SphericalParticleContactForce
  {
  public:
    explicit SphericalParticleContactForce(
      const SphericalParticleContactData<number> &contact_data);

    /**
     * Compute the contact forces and add them to the obstacles in the given obstacle field. This
     * algorithm consists of two steps, first finding contacting particles, then computing and
     * adding the contact forces.
     *
     * For finding contacting particles, a brute-force approach is used where each particle is
     * checked against all other particles in the global particle data structure.
     *
     * For the contact force computation, a nonlinear spring-dashpot model (Hertz contact theory) is
     * used to compute normal contact forces between spherical particles. Currently only normal
     * contact forces are computed; tangential forces and friction are not yet implemented. For
     * details on the implemented model, see e.g. Gaboriault et al. (DOI:10.48550/arXiv.2509.26402).
     *
     * @param obstacle_field The obstacle field containing the particles for which contact forces are to be
     * computed. The resulting forces are added directly to the particles in this field.
     */
    void
    add_load_to_obstacles(ObstacleField<dim, number, ObstacleType> &obstacle_field) const;

  private:
    /// Contact data for the spherical particle contact force model.
    const SphericalParticleContactData<number> &contact_data;

    /// Damping prefactor computed from the restitution coefficient. This is cached for efficiency.
    const number damping_prefactor;

    /**
     * Struct describing the contact configuration between two particles, i.e., it computes and
     * caches all relevant contact parameters such as contact velocity, overlaps etc. required in
     * the computation of contact forces.
     */
    struct ContactConfiguration
    {
      /**
       * Constructor which computes all relevant input for the contact model between two particles.
       * Note that the computed normal vector points from @p self to @p other.
       *
       * @param self The first particle in the contact.
       * @param other The second particle in the contact.
       * @param youngs_modulus The Young's modulus of the particle material.
       * @param poisson_ratio The Poisson's ratio of the particle material.
       */
      ContactConfiguration(DEMParticleAccessor<dim, number> &self,
                           DEMParticleAccessor<dim, number> &other,
                           const number                      youngs_modulus,
                           const number                      poisson_ratio);

      /// Effective mass of the two contacting particles, i.e., m1*m2/(m1+m2).
      number effective_mass;

      /// Effective radius of the two contacting particles, i.e., R1*R2/(R1+R2).
      number effective_radius;

      /// Effective youngs modulus of the two contacting particles, i.e., E/(2*(1-nu^2)).
      number effective_youngs_modulus;

      /// Normal vector pointing from self to other.
      dealii::Tensor<1, dim, number> normal_vector;

      /// Normal overlap between the two particles (positive if in contact).
      number normal_overlap;

      /**
       * Struct for describing the relative velocity between two particles at the contact point.
       */
      struct
      {
        /// Full relative velocity vector at the contact point.
        dealii::Tensor<1, dim, number> value;

        /// Normal component of the relative velocity.
        dealii::Tensor<1, dim, number> normal_component;

        /// Tangential component of the relative velocity.
        dealii::Tensor<1, dim, number> tangential_component;
      } relative_velocity;
    };

    /**
     * Computes the normal contact force for a given particle–particle contact. The formulation
     * follows Gaboriault et al. (DOI:10.48550/arXiv.2509.26402).
     *
     * The normal contact force is defined as
     * \f[
     * \boldsymbol{F}_n = k_n\delta_n\boldsymbol{n} + \eta_n\boldsymbol{v}_n,
     * \f]
     * where \f$k_n\f$ is the normal stiffness, \f$\delta_n\f$ the normal overlap,
     * \f$\boldsymbol{n}\f$ the unit normal vector at the contact, \f$\eta_n\f$ the normal damping
     * coefficient, and \f$v_n\f$ the normal component of the relative velocity.
     *
     * The normal stiffness is computed as
     * \f[
     * k_n = \frac{4}{3} Y_e \sqrt{R_e \delta_n},
     * \f]
     * where \f$Y_e\f$ is the effective Young's modulus and \f$R_e\f$ the effective radius. The
     * normal damping coefficient is given by
     * \f[
     * \eta_n = c_d \sqrt{\frac{3}{2} k_n m_e},
     * \f]
     * with \f$c_d\f$ being a damping prefactor derived from the coefficient of restitution (see
     * compute_damping_prefactor()) and \f$m_e\f$ the effective mass.
     *
     * @param contact_configuration  Configuration of the contact between two particles.
     * @return The computed normal contact force vector.
     */

    dealii::Tensor<1, dim, number>
    normal_contact_force(const ContactConfiguration &contact_configuration) const;

    /**
     * Function which computes the damping prefactor from the restitution coefficient
     * \f[
     * \eta_t = 2 \sqrt{\frac{5}{6}}\frac{\ln(e_r)}{\sqrt{\ln^2(e_r) + \pi^2}},
     * \f]
     * with \f$e_r\f$ being the coefficient of restitution.
     *
     * @param restitution_coefficient The coefficient of restitution.
     * @return The computed damping prefactor.
     */
    number
    compute_damping_prefactor(const number restitution_coefficient) const;
  };
} // namespace MeltPoolDG