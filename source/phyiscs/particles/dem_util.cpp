#include <deal.II/base/exceptions.h>

#include <meltpooldg/particles/dem_util.hpp>

#include <iterator>

template <int dim, typename number>
MeltPoolDG::ParticleHandlerVectorView<dim, number>::ParticleHandlerVectorView(
  dealii::Particles::ParticleHandler<dim> &particle_handler,
  const unsigned                           property_index,
  const bool                               index_describes_location)
  : particle_handler(particle_handler)
  , property_index(property_index)
  , index_describes_location(index_describes_location)
{}

template <int dim, typename number>
MeltPoolDG::ParticleHandlerVectorView<dim, number> &
MeltPoolDG::ParticleHandlerVectorView<dim, number>::operator*=(const number factor)
{
  for (unsigned i = 0; i < locally_owned_size(); ++i)
    local_element(i) *= factor;
  return *this;
}

template <int dim, typename number>
MeltPoolDG::ParticleHandlerVectorView<dim, number> &
MeltPoolDG::ParticleHandlerVectorView<dim, number>::operator/=(const number factor)
{
  for (unsigned i = 0; i < locally_owned_size(); ++i)
    local_element(i) /= factor;
  return *this;
}

template <int dim, typename number>
MeltPoolDG::ParticleHandlerVectorView<dim, number> &
MeltPoolDG::ParticleHandlerVectorView<dim, number>::operator-=(
  const ParticleHandlerVectorView<dim, number> &v)
{
  AssertDimension(locally_owned_size(), v.locally_owned_size());
  for (unsigned i = 0; i < locally_owned_size(); ++i)
    local_element(i) -= v.local_element(i);
  return *this;
}

template <int dim, typename number>
MeltPoolDG::ParticleHandlerVectorView<dim, number> &
MeltPoolDG::ParticleHandlerVectorView<dim, number>::operator+=(
  const ParticleHandlerVectorView<dim, number> &v)
{
  AssertDimension(locally_owned_size(), v.locally_owned_size());
  for (unsigned i = 0; i < locally_owned_size(); ++i)
    local_element(i) -= v.local_element(i);
  return *this;
}

template <int dim, typename number>
void
MeltPoolDG::ParticleHandlerVectorView<dim, number>::add(
  const number                                  a,
  const ParticleHandlerVectorView<dim, number> &v)
{
  AssertDimension(locally_owned_size(), v.locally_owned_size());

  for (unsigned i = 0; i < locally_owned_size(); ++i)
    local_element(i) += a * v.local_element(i);
}

template <int dim, typename number>
void
MeltPoolDG::ParticleHandlerVectorView<dim, number>::sadd(
  const number                                  s,
  const number                                  a,
  const ParticleHandlerVectorView<dim, number> &v)
{
  AssertDimension(locally_owned_size(), v.locally_owned_size());

  for (unsigned i = 0; i < locally_owned_size(); ++i)
    {
      local_element(i) *= s;
      local_element(i) += a * v.local_element(i);
    }
}

template <int dim, typename number>
bool
MeltPoolDG::ParticleHandlerVectorView<dim, number>::has_ghost_elements() const
{
  return true;
}

template <int dim, typename number>
void
MeltPoolDG::ParticleHandlerVectorView<dim, number>::zero_out_ghost_elements() const
{
  // do nothing
}

template <int dim, typename number>
void
MeltPoolDG::ParticleHandlerVectorView<dim, number>::update_ghost_values(
  const bool exchange_particles) const
{
  if (exchange_particles)
    particle_handler.exchange_ghost_particles(true);
  particle_handler.update_ghost_particles();
}

template <int dim, typename number>
unsigned
MeltPoolDG::ParticleHandlerVectorView<dim, number>::locally_owned_size() const
{
  return particle_handler.n_locally_owned_particles();
}

template <int dim, typename number>
number &
MeltPoolDG::ParticleHandlerVectorView<dim, number>::local_element(unsigned index)
{
  if (index_describes_location)
    return std::next(particle_handler.begin(), index)->get_location()[property_index];
  else
    return std::next(particle_handler.begin(), index)->get_properties()[property_index];
}

template <int dim, typename number>
const number &
MeltPoolDG::ParticleHandlerVectorView<dim, number>::local_element(unsigned index) const
{
  if (index_describes_location)
    return std::next(particle_handler.begin(), index)->get_location()[property_index];
  else
    return std::next(particle_handler.begin(), index)->get_properties()[property_index];
}

template <int dim, typename number>
unsigned
MeltPoolDG::ParticleHandlerVectorView<dim, number>::size() const
{
  return particle_handler.n_global_particles();
}

template <int dim, typename number>
void
MeltPoolDG::ParticleHandlerBlockVectorView<dim, number>::reinit(
  dealii::Particles::ParticleHandler<dim> &particle_handler)
{
  vectors.reserve(dim);
  for (int i = 0; i < dim; ++i)
    vectors.emplace_back(ParticleHandlerVectorView<dim, number>(particle_handler, i, true));
}

template <int dim, typename number>
void
MeltPoolDG::ParticleHandlerBlockVectorView<dim, number>::reinit(
  dealii::Particles::ParticleHandler<dim> &particle_handler,
  const std::vector<unsigned>             &property_indices)
{
  vectors.reserve(property_indices.size());
  for (auto idx : property_indices)
    vectors.emplace_back(ParticleHandlerVectorView<dim, number>(particle_handler, idx, false));
}

template <int dim, typename number>
void
MeltPoolDG::ParticleHandlerBlockVectorView<dim, number>::reinit(
  dealii::Particles::ParticleHandler<dim> &particle_handler,
  const unsigned                           property_indices_begin,
  const unsigned                           property_indices_size)
{
  vectors.reserve(property_indices_size);
  for (unsigned idx = property_indices_begin; idx < property_indices_begin + property_indices_size;
       ++idx)
    vectors.emplace_back(ParticleHandlerVectorView<dim, number>(particle_handler, idx, false));
}

template <int dim, typename number>
bool
MeltPoolDG::ParticleHandlerBlockVectorView<dim, number>::has_ghost_elements() const
{
  for (auto &elem : vectors)
    if (elem.has_ghost_elements())
      return true;
  return false;
}

template <int dim, typename number>
void
MeltPoolDG::ParticleHandlerBlockVectorView<dim, number>::zero_out_ghost_elements() const
{
  for (auto &elem : vectors)
    elem.zero_out_ghost_elements();
}

template <int dim, typename number>
void
MeltPoolDG::ParticleHandlerBlockVectorView<dim, number>::update_ghost_values(
  const bool exchange_particles) const
{
  for (auto &elem : vectors)
    elem.update_ghost_values(exchange_particles);
}

template <int dim, typename number>
void
MeltPoolDG::ParticleHandlerBlockVectorView<dim, number>::add(
  const number                                       a,
  const ParticleHandlerBlockVectorView<dim, number> &v)
{
  AssertDimension(n_blocks(), v.n_blocks());
  for (unsigned i = 0; i < n_blocks(); ++i)
    vectors[i].add(a, v.block(i));
}

template <int dim, typename number>
void
MeltPoolDG::ParticleHandlerBlockVectorView<dim, number>::sadd(
  const number                                       s,
  const number                                       a,
  const ParticleHandlerBlockVectorView<dim, number> &v)
{
  AssertDimension(n_blocks(), v.n_blocks());
  for (unsigned i = 0; i < n_blocks(); ++i)
    vectors[i].sadd(s, a, v.block(i));
}

template <int dim, typename number>
unsigned
MeltPoolDG::ParticleHandlerBlockVectorView<dim, number>::n_blocks() const
{
  return vectors.size();
}

template <int dim, typename number>
const MeltPoolDG::ParticleHandlerVectorView<dim, number> &
MeltPoolDG::ParticleHandlerBlockVectorView<dim, number>::block(const unsigned i) const
{
  return vectors[i];
}

template <int dim, typename number>
MeltPoolDG::ParticleHandlerVectorView<dim, number> &
MeltPoolDG::ParticleHandlerBlockVectorView<dim, number>::block(const unsigned i)
{
  return vectors[i];
}

template class MeltPoolDG::ParticleHandlerVectorView<1, double>;
template class MeltPoolDG::ParticleHandlerVectorView<2, double>;
template class MeltPoolDG::ParticleHandlerVectorView<3, double>;

template class MeltPoolDG::ParticleHandlerBlockVectorView<1, double>;
template class MeltPoolDG::ParticleHandlerBlockVectorView<2, double>;
template class MeltPoolDG::ParticleHandlerBlockVectorView<3, double>;