#include <deal.II/base/exceptions.h>

#include "meltpooldg/particles/obstacle_data_structure.hpp"
#include "meltpooldg/particles/particle.hpp"
#include <meltpooldg/particles/dem_util.hpp>

#include <iterator>

template <int dim, typename number, typename DemDataStructure>
MeltPoolDG::ParticleHandlerVectorView<dim, number, DemDataStructure>::ParticleHandlerVectorView(
  DemDataStructure &particle_handler,
  const unsigned    property_index,
  const bool        index_describes_location)
  : particle_handler(particle_handler)
  , property_index(property_index)
  , index_describes_location(index_describes_location)
{}

template <int dim, typename number, typename DemDataStructure>
MeltPoolDG::ParticleHandlerVectorView<dim, number, DemDataStructure> &
MeltPoolDG::ParticleHandlerVectorView<dim, number, DemDataStructure>::operator*=(
  const number factor)
{
  for (unsigned i = 0; i < locally_owned_size(); ++i)
    local_element(i) *= factor;
  return *this;
}

template <int dim, typename number, typename DemDataStructure>
MeltPoolDG::ParticleHandlerVectorView<dim, number, DemDataStructure> &
MeltPoolDG::ParticleHandlerVectorView<dim, number, DemDataStructure>::operator/=(
  const number factor)
{
  for (unsigned i = 0; i < locally_owned_size(); ++i)
    local_element(i) /= factor;
  return *this;
}

template <int dim, typename number, typename DemDataStructure>
MeltPoolDG::ParticleHandlerVectorView<dim, number, DemDataStructure> &
MeltPoolDG::ParticleHandlerVectorView<dim, number, DemDataStructure>::operator-=(
  const ParticleHandlerVectorView<dim, number, DemDataStructure> &v)
{
  AssertDimension(locally_owned_size(), v.locally_owned_size());
  for (unsigned i = 0; i < locally_owned_size(); ++i)
    local_element(i) -= v.local_element(i);
  return *this;
}

template <int dim, typename number, typename DemDataStructure>
MeltPoolDG::ParticleHandlerVectorView<dim, number, DemDataStructure> &
MeltPoolDG::ParticleHandlerVectorView<dim, number, DemDataStructure>::operator+=(
  const ParticleHandlerVectorView<dim, number, DemDataStructure> &v)
{
  AssertDimension(locally_owned_size(), v.locally_owned_size());
  for (unsigned i = 0; i < locally_owned_size(); ++i)
    local_element(i) -= v.local_element(i);
  return *this;
}

template <int dim, typename number, typename DemDataStructure>
void
MeltPoolDG::ParticleHandlerVectorView<dim, number, DemDataStructure>::add(
  const number                                                    a,
  const ParticleHandlerVectorView<dim, number, DemDataStructure> &v)
{
  AssertDimension(locally_owned_size(), v.locally_owned_size());

  for (unsigned i = 0; i < locally_owned_size(); ++i)
    local_element(i) += a * v.local_element(i);
}

template <int dim, typename number, typename DemDataStructure>
void
MeltPoolDG::ParticleHandlerVectorView<dim, number, DemDataStructure>::sadd(
  const number                                                    s,
  const number                                                    a,
  const ParticleHandlerVectorView<dim, number, DemDataStructure> &v)
{
  AssertDimension(locally_owned_size(), v.locally_owned_size());

  for (unsigned i = 0; i < locally_owned_size(); ++i)
    {
      local_element(i) *= s;
      local_element(i) += a * v.local_element(i);
    }
}

template <int dim, typename number, typename DemDataStructure>
bool
MeltPoolDG::ParticleHandlerVectorView<dim, number, DemDataStructure>::has_ghost_elements() const
{
  return true;
}

template <int dim, typename number, typename DemDataStructure>
void
MeltPoolDG::ParticleHandlerVectorView<dim, number, DemDataStructure>::zero_out_ghost_elements()
  const
{
  // do nothing
}

template <int dim, typename number, typename DemDataStructure>
void
MeltPoolDG::ParticleHandlerVectorView<dim, number, DemDataStructure>::update_ghost_values()
{
  particle_handler.update_ghost_particles();
}

template <int dim, typename number, typename DemDataStructure>
unsigned
MeltPoolDG::ParticleHandlerVectorView<dim, number, DemDataStructure>::locally_owned_size() const
{
  return particle_handler.n_locally_owned_particles();
}

template <int dim, typename number, typename DemDataStructure>
number &
MeltPoolDG::ParticleHandlerVectorView<dim, number, DemDataStructure>::local_element(unsigned index)
{
  if (index_describes_location)
    return std::next(particle_handler.begin(), index)->get_location()[property_index];
  else
    return std::next(particle_handler.begin(), index)->get_properties()[property_index];
}

template <int dim, typename number, typename DemDataStructure>
const number &
MeltPoolDG::ParticleHandlerVectorView<dim, number, DemDataStructure>::local_element(
  unsigned index) const
{
  if (index_describes_location)
    return std::next(particle_handler.begin(), index)->get_location()[property_index];
  else
    return std::next(particle_handler.begin(), index)->get_properties()[property_index];
}

template <int dim, typename number, typename DemDataStructure>
unsigned
MeltPoolDG::ParticleHandlerVectorView<dim, number, DemDataStructure>::size() const
{
  return particle_handler.n_global_particles();
}

template <int dim, typename number, typename DemDataStructure>
void
MeltPoolDG::ParticleHandlerBlockVectorView<dim, number, DemDataStructure>::reinit(
  DemDataStructure &particle_handler)
{
  vectors.reserve(dim);
  for (int i = 0; i < dim; ++i)
    vectors.emplace_back(
      ParticleHandlerVectorView<dim, number, DemDataStructure>(particle_handler, i, true));
}

template <int dim, typename number, typename DemDataStructure>
void
MeltPoolDG::ParticleHandlerBlockVectorView<dim, number, DemDataStructure>::reinit(
  DemDataStructure            &particle_handler,
  const std::vector<unsigned> &property_indices)
{
  vectors.reserve(property_indices.size());
  for (auto idx : property_indices)
    vectors.emplace_back(
      ParticleHandlerVectorView<dim, number, DemDataStructure>(particle_handler, idx, false));
}

template <int dim, typename number, typename DemDataStructure>
void
MeltPoolDG::ParticleHandlerBlockVectorView<dim, number, DemDataStructure>::reinit(
  DemDataStructure &particle_handler,
  const unsigned    property_indices_begin,
  const unsigned    property_indices_size)
{
  vectors.reserve(property_indices_size);
  for (unsigned idx = property_indices_begin; idx < property_indices_begin + property_indices_size;
       ++idx)
    vectors.emplace_back(
      ParticleHandlerVectorView<dim, number, DemDataStructure>(particle_handler, idx, false));
}

template <int dim, typename number, typename DemDataStructure>
bool
MeltPoolDG::ParticleHandlerBlockVectorView<dim, number, DemDataStructure>::has_ghost_elements()
  const
{
  for (auto &elem : vectors)
    if (elem.has_ghost_elements())
      return true;
  return false;
}

template <int dim, typename number, typename DemDataStructure>
void
MeltPoolDG::ParticleHandlerBlockVectorView<dim, number, DemDataStructure>::zero_out_ghost_elements()
  const
{
  for (auto &elem : vectors)
    elem.zero_out_ghost_elements();
}

template <int dim, typename number, typename DemDataStructure>
void
MeltPoolDG::ParticleHandlerBlockVectorView<dim, number, DemDataStructure>::update_ghost_values()
{
  for (auto &elem : vectors)
    elem.update_ghost_values();
}

template <int dim, typename number, typename DemDataStructure>
void
MeltPoolDG::ParticleHandlerBlockVectorView<dim, number, DemDataStructure>::add(
  const number                                                         a,
  const ParticleHandlerBlockVectorView<dim, number, DemDataStructure> &v)
{
  AssertDimension(n_blocks(), v.n_blocks());
  for (unsigned i = 0; i < n_blocks(); ++i)
    vectors[i].add(a, v.block(i));
}

template <int dim, typename number, typename DemDataStructure>
void
MeltPoolDG::ParticleHandlerBlockVectorView<dim, number, DemDataStructure>::sadd(
  const number                                                         s,
  const number                                                         a,
  const ParticleHandlerBlockVectorView<dim, number, DemDataStructure> &v)
{
  AssertDimension(n_blocks(), v.n_blocks());
  for (unsigned i = 0; i < n_blocks(); ++i)
    vectors[i].sadd(s, a, v.block(i));
}

template <int dim, typename number, typename DemDataStructure>
unsigned
MeltPoolDG::ParticleHandlerBlockVectorView<dim, number, DemDataStructure>::n_blocks() const
{
  return vectors.size();
}

template <int dim, typename number, typename DemDataStructure>
const MeltPoolDG::ParticleHandlerVectorView<dim, number, DemDataStructure> &
MeltPoolDG::ParticleHandlerBlockVectorView<dim, number, DemDataStructure>::block(
  const unsigned i) const
{
  return vectors[i];
}

template <int dim, typename number, typename DemDataStructure>
MeltPoolDG::ParticleHandlerVectorView<dim, number, DemDataStructure> &
MeltPoolDG::ParticleHandlerBlockVectorView<dim, number, DemDataStructure>::block(const unsigned i)
{
  return vectors[i];
}

template class MeltPoolDG::ParticleHandlerVectorView<
  1,
  double,
  MeltPoolDG::ObstacleCompleteDomainSearch<1, double, MeltPoolDG::SphericalParticle<1, double>>>;
template class MeltPoolDG::ParticleHandlerVectorView<
  2,
  double,
  MeltPoolDG::ObstacleCompleteDomainSearch<2, double, MeltPoolDG::SphericalParticle<2, double>>>;
template class MeltPoolDG::ParticleHandlerVectorView<
  3,
  double,
  MeltPoolDG::ObstacleCompleteDomainSearch<3, double, MeltPoolDG::SphericalParticle<3, double>>>;

template class MeltPoolDG::ParticleHandlerBlockVectorView<
  1,
  double,
  MeltPoolDG::ObstacleCompleteDomainSearch<1, double, MeltPoolDG::SphericalParticle<1, double>>>;
template class MeltPoolDG::ParticleHandlerBlockVectorView<
  2,
  double,
  MeltPoolDG::ObstacleCompleteDomainSearch<2, double, MeltPoolDG::SphericalParticle<2, double>>>;
template class MeltPoolDG::ParticleHandlerBlockVectorView<
  3,
  double,
  MeltPoolDG::ObstacleCompleteDomainSearch<3, double, MeltPoolDG::SphericalParticle<3, double>>>;