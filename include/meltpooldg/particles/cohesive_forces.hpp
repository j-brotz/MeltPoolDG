#pragma once

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/tensor.h>

#include <meltpooldg/particles/cohesive_forces_data.hpp>
#include <meltpooldg/particles/obstacle_field.hpp>
#include <meltpooldg/particles/particle_accessor.hpp>

namespace MeltPoolDG
{
  template <int dim, typename number, typename ObstacleType>
  class SphericalParticleCohesiveForce
  {
  public:
    /**
     * Constructor for the spherical particle cohesive force model.
     *
     * @param cohesive_force_data Cohesive force data defining the material and cohesive
     * contact properties.
     */
    explicit SphericalParticleCohesiveForce(
      const SphericalParticleCohesiveForceData<number> &cohesive_force_data);

    /**
     * Computes the total cohesive load acting on all particles in a given obstacle field. The
     * formulation follows Meier et al. (DOI:10.1016/j.powtec.2018.11.072). The resulting forces are
     * accumulated and directly applied to the particles contained in the obstacle field.
     *
     * For the computation of the cohesive force between two particles, three interaction regimes
     * are distinguished based on their surface-to-surface distance.
     *
     * For particles in close contact, i.e., if the surface-to-surface distance is smaller than
     * \f$g_0\f$, a constant pull-off force based on the DMT model is applied:
     * \f[
     * \boldsymbol{F}_{\mathrm{po}} = 4 \pi R_e \gamma \boldsymbol{N},
     * \f]
     * where \f$R_e\f$ is the effective radius, \f$\gamma\f$ the surface energy, and
     * \f$\boldsymbol{N}\f$ the unit normal vector pointing from one particle to the other. The
     * characteristic distance \f$g_0\f$ is given by
     * \f[
     * g_0 = \sqrt{\frac{A R_e}{6 \lVert \boldsymbol{F}_{\mathrm{po}} \rVert}}.
     * \f]
     *
     * If the surface-to-surface distance is larger than \f$g_0\f$ but smaller than the cut-off
     * distance \f$g^*\f$, the cohesive force is computed as
     * \f[
     * \boldsymbol{F} = \frac{A R_e}{6 g_N^2} \boldsymbol{N},
     * \f]
     * where \f$A\f$ is the Hamaker constant and \f$g_N\f$ the surface-to-surface distance. The
     * cut-off distance is defined as
     * \f[
     * g^* = \frac{g_0}{\sqrt{c_{\mathrm{FPO}}}},
     * \f]
     * where \f$c_{\mathrm{FPO}}\f$ denotes the relative cut-off decline of the van der Waals force.
     *
     * For surface-to-surface distances larger than the cut-off distance \f$g^*\f$, no cohesive
     * forces are applied.
     *
     * @param obstacle_field  Obstacle field containing the particles for which cohesive forces are
     *  computed. The resulting forces are added directly to the particles in this field.
     */
    void
    add_load_to_obstacles(ObstacleField<dim, number, ObstacleType> &obstacle_field) const;

  private:
    /// Cohesive force data for the spherical particle cohesive force model.
    SphericalParticleCohesiveForceData<number> cohesive_force_data;

    /**
     * Struct describing the cohesive contact configuration between two particles, i.e., it computes
     * and caches all relevant cohesive contact parameters such as effective radius, cut-off
     * distances etc. required in the computation of cohesive forces.
     */
    struct CohesiveContactConfiguration
    {
      /**
       * Constructor which computes all relevant input for the cohesive force model between two
       * particles. Note that the computed normal vector points from @p self to @p other.
       *
       * @param self The particle for which the cohesive force is computed.
       * @param other  The other particle in the cohesive contact.
       * @param cohesive_force_data The cohesive force data defining the material and cohesive contact
       * properties.
       */
      CohesiveContactConfiguration(
        const DEMParticleAccessor<dim, number>           &self,
        const DEMParticleAccessor<dim, number>           &other,
        const SphericalParticleCohesiveForceData<number> &cohesive_force_data);

      /// Normal distance between the two particles (positive if not in contact), i.e., the smallest
      /// distance between their surfaces.
      number normal_distance;

      /// Effective radius of the two particles, i.e., R1*R2/(R1+R2).
      number effective_radius;

      /// Magnitude of the pull-off force according to the DMT model.
      number pull_off_force_magnitude;

      /// Cut-off distance beyond which no van der Waals forces are considered, i.e., beyond which
      /// the force is set to zero.
      number cut_off_distance; // denoted as g* in literature

      /// Distance below which the pull-off force is applied.
      number pull_off_force_limit_distance; // denoted as g_0 in literature

      /// Unit normal vector pointing from self to other.
      dealii::Tensor<1, dim, number> normal_vector;
    };
  };

} // namespace MeltPoolDG
