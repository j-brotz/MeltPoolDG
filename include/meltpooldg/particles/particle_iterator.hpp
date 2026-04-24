
#pragma once

#include <deal.II/base/exception_macros.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/types.h>

#include <deal.II/particles/particle_accessor.h>
#include <deal.II/particles/property_pool.h>

#include <meltpooldg/particles/dem_util.hpp>
#include <meltpooldg/particles/particle.hpp>
#include <meltpooldg/particles/particle_accessor.hpp>

#include <cstddef>
#include <iterator>
#include <memory>
#include <optional>
#include <type_traits>
#include <variant>

namespace MeltPoolDG
{
  /**
   * @brief STL-style bidirectional iterator over particles.
   *
   * This iterator provides a uniform interface to traverse particles, whether they belong to a
   * `dealii::Particles::ParticleHandler` or are stored in a `PropertyPool`. The interface is
   * consistent for both cases and supports standard STL-style iterator operations, including:
   * `operator*`, `operator->`, `operator++`, `operator--`, and comparison operators.
   *
   * Dereferencing the iterator yields a `ParticleAccessor` object, which grants direct access
   * to the particle's properties as defined in MeltPoolDG.
   */
  template <int dim, typename number>
  class ParticleIterator
  {
  public:
    using iterator_category = std::bidirectional_iterator_tag;
    using difference_type   = std::ptrdiff_t;
    using value_type        = DEMParticleAccessor<dim, number>;
    using reference         = DEMParticleAccessor<dim, number> &;
    using pointer           = DEMParticleAccessor<dim, number> *;

    /**
     * Default constructor; creates an invalid iterator.
     */
    ParticleIterator() = default;

    /**
     * Construct the iterator from a deal.II particle iterator.
     *
     * @param particle The underlying deal.II particle iterator.
     */
    ParticleIterator(dealii::Particles::ParticleIterator<dim> particle);

    /**
     * Construct from a deal.II particle property pool and a corresponding handle identifying the
     * particle of interest in the property pool.
     *
     * @param property_pool The property pool containing the particle.
     * @param handle Handle of the particle in the property pool.
     */
    ParticleIterator(dealii::Particles::PropertyPool<dim>                       &property_pool,
                     const typename dealii::Particles::PropertyPool<dim>::Handle handle);

    /**
     * Dereference operator.
     *
     * @throws Assert if the iterator is invalid (only in debug mode).
     */
    DEMParticleAccessor<dim, number> &
    operator*();

    /**
     * Same as above but const.
     */
    const DEMParticleAccessor<dim, number> &
    operator*() const;

    /**
     * Arrow operator.
     *
     * @throws Assert if the iterator is invalid (only in debug mode).
     */
    DEMParticleAccessor<dim, number> *
    operator->();

    /**
     * Same as above but const.
     */
    const DEMParticleAccessor<dim, number> *
    operator->() const;

    [[nodiscard]] bool
    operator!=(const ParticleIterator &c) const;

    [[nodiscard]] bool
    operator==(const ParticleIterator &c) const;

    /**
     * Pre-increment operator. Advances to the next particle.
     */
    ParticleIterator &
    operator++();

    /**
     * Post-increment operator.
     */
    ParticleIterator
    operator++(int);

    /**
     * Pre-decrement operator. Moves to the previous particle.
     */
    ParticleIterator &
    operator--();

    /**
     * Post-decrement operator.
     */
    ParticleIterator
    operator--(int);

  private:
    /// Cached particle accessor for the current iterator position. Note the optional will be empty
    /// in the case that the iterator points to an invalid position.
    std::optional<DEMParticleAccessor<dim, number>> particle_accessor;

    /**
     * Iterator data structure for iterating over a deal.II property pool. This struct allows to
     * iterate over all particles stored in a property pool without actually having a corresponding
     * deal.II particle handler object.
     */
    struct PropertyPoolIteratorData
    {
      dealii::Particles::PropertyPool<dim>                 *dealii_property_pool = nullptr;
      typename dealii::Particles::PropertyPool<dim>::Handle handle =
        dealii::numbers::invalid_unsigned_int;

      void
      operator++() noexcept
      {
        ++handle;
      }

      void
      operator--() noexcept
      {
        --handle;
      }

      [[nodiscard]] bool
      operator==(const PropertyPoolIteratorData &c) const noexcept
      {
        return handle == c.handle and dealii_property_pool == c.dealii_property_pool;
      }

      [[nodiscard]] bool
      operator!=(const PropertyPoolIteratorData &c) const noexcept
      {
        return handle != c.handle or dealii_property_pool != c.dealii_property_pool;
      }

      dealii::IteratorState::IteratorStates
      state() const noexcept
      {
        if (dealii_property_pool == nullptr or handle >= dealii_property_pool->n_registered_slots())
          return dealii::IteratorState::IteratorStates::invalid;
        else
          return dealii::IteratorState::IteratorStates::valid;
      }
    };

    /// Variant type for the underlying iterator.
    std::variant<dealii::Particles::ParticleIterator<dim>, PropertyPoolIteratorData>
      deal_ii_iterator;

    /**
     * Construct the particle accessor from the current iterator state.
     *
     * @note Does not check whether the iterator is valid; use refresh_accessor() for that.
     */
    void
    make_accessor();

    /**
     * Refreshes the internal accessor to match the current iterator state. Sets the cached
     * `particle_accessor` or resets it if the iterator is invalid.
     */
    void
    refresh_accessor();
  };


  template <int dim, typename number>
  ParticleIterator<dim, number>::ParticleIterator(dealii::Particles::ParticleIterator<dim> particle)
    : deal_ii_iterator(particle)
  {
    refresh_accessor();
  }

  template <int dim, typename number>
  ParticleIterator<dim, number>::ParticleIterator(
    dealii::Particles::PropertyPool<dim>                       &property_pool,
    const typename dealii::Particles::PropertyPool<dim>::Handle handle)
    : deal_ii_iterator(PropertyPoolIteratorData{&property_pool, handle})
  {
    refresh_accessor();
  }

  template <int dim, typename number>
  DEMParticleAccessor<dim, number> &
  ParticleIterator<dim, number>::operator*()
  {
    Assert(particle_accessor.has_value(),
           dealii::ExcMessage("Dereferencing invalid ParticleIterator"));
    return *particle_accessor;
  }

  template <int dim, typename number>
  const DEMParticleAccessor<dim, number> &
  ParticleIterator<dim, number>::operator*() const
  {
    Assert(particle_accessor.has_value(),
           dealii::ExcMessage("Dereferencing invalid ParticleIterator"));
    return *particle_accessor;
  }

  template <int dim, typename number>
  DEMParticleAccessor<dim, number> *
  ParticleIterator<dim, number>::operator->()
  {
    Assert(particle_accessor.has_value(),
           dealii::ExcMessage("Dereferencing invalid ParticleIterator"));
    return std::addressof(**this);
  }

  template <int dim, typename number>
  const DEMParticleAccessor<dim, number> *
  ParticleIterator<dim, number>::operator->() const
  {
    Assert(particle_accessor.has_value(),
           dealii::ExcMessage("Dereferencing invalid ParticleIterator"));
    return std::addressof(**this);
  }

  template <int dim, typename number>
  [[nodiscard]] bool
  ParticleIterator<dim, number>::operator==(const ParticleIterator &c) const
  {
    return deal_ii_iterator == c.deal_ii_iterator;
  }

  template <int dim, typename number>
  [[nodiscard]] bool
  ParticleIterator<dim, number>::operator!=(const ParticleIterator &c) const
  {
    return deal_ii_iterator != c.deal_ii_iterator;
  }

  template <int dim, typename number>
  ParticleIterator<dim, number> &
  ParticleIterator<dim, number>::operator++()
  {
    std::visit([&](auto &it) { ++it; }, deal_ii_iterator);
    refresh_accessor();
    return *this;
  }

  template <int dim, typename number>
  ParticleIterator<dim, number>
  ParticleIterator<dim, number>::operator++(int)
  {
    ParticleIterator tmp(*this);
                     operator++();
    return tmp;
  }

  template <int dim, typename number>
  ParticleIterator<dim, number> &
  ParticleIterator<dim, number>::operator--()
  {
    std::visit([&](auto &it) { --it; }, deal_ii_iterator);
    refresh_accessor();
    return *this;
  }

  template <int dim, typename number>
  ParticleIterator<dim, number>
  ParticleIterator<dim, number>::operator--(int)
  {
    ParticleIterator tmp(*this);
                     operator--();
    return tmp;
  }

  template <int dim, typename number>
  void
  ParticleIterator<dim, number>::make_accessor()
  {
    std::visit(
      [&](auto &it) {
        using T = std::decay_t<decltype(it)>;

        if constexpr (std::is_same_v<T, dealii::Particles::ParticleIterator<dim>>)
          particle_accessor.emplace(*it);
        else if constexpr (std::is_same_v<T, PropertyPoolIteratorData>)
          particle_accessor.emplace(*it.dealii_property_pool, it.handle);
        else
          AssertThrow(false, dealii::ExcInternalError());
      },
      deal_ii_iterator);
  }

  template <int dim, typename number>
  void
  ParticleIterator<dim, number>::refresh_accessor()
  {
    std::visit(
      [&](auto &it) {
        if (it.state() == dealii::IteratorState::IteratorStates::valid)
          make_accessor();
        else
          particle_accessor = std::nullopt;
      },
      deal_ii_iterator);
  }
} // namespace MeltPoolDG
