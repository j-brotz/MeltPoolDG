import numpy as np
from scipy.stats import truncnorm


def truncated_normal(n, vmin, vmax):
    """
    Sample values from a truncated normal distribution.

    The distribution is centered at the midpoint of [vmin, vmax] and
    scaled so that ~99.7% of values would lie within the interval before truncation.

    Parameters
    ----------
    n : int
        Number of samples to generate.
    vmin : float
        Minimum allowed value.
    vmax : float
        Maximum allowed value.

    Returns
    -------
    np.ndarray
        Array of shape (n,) containing truncated normal samples.
    """

    mean = 0.5 * (vmin + vmax)
    std = (vmax - vmin) / 6.0

    a = (vmin - mean) / std
    b = (vmax - mean) / std

    return truncnorm.rvs(a, b, loc=mean, scale=std, size=n)


def generate_particles(
    n_particles,
    dim,
    box_min,
    box_max,
    radius_range,
    density_range,
    max_attempts=10000,
    seed=None,
):
    """
    Generate random particle configuration with truncated Gaussian sampling.

    Output format:
    position_0, position_1, ..., position_{dim-1}, density, radius
    """
    if seed is not None:
        np.random.seed(seed)

    # Convert domain bounds to arrays for vector operations
    box_min = np.array(box_min[:dim])
    box_max = np.array(box_max[:dim])

    # Pre-sample physical properties
    radii = truncated_normal(n_particles, *radius_range)
    densities = truncated_normal(n_particles, *density_range)

    positions = []
    accepted_radii = []
    accepted_densities = []

    # Place particles one by one
    for i in range(n_particles):
        r = radii[i]
        d = densities[i]

        placed = False

        # Try random placements until a valid one is found
        for _ in range(max_attempts):

            # Ensure particle stays inside the box (account for radius)
            pos = np.random.uniform(low=box_min + r, high=box_max - r)

            # Check for overlaps with previously placed particles
            overlap = False
            for j, p_old in enumerate(positions):
                dist = np.linalg.norm(pos - p_old)
                if dist < (r + accepted_radii[j]):
                    overlap = True
                    break

            # Accept position if no overlap
            if not overlap:
                positions.append(pos)
                accepted_radii.append(r)
                accepted_densities.append(d)
                placed = True
                break

        # Fail safely if placement is impossible
        if not placed:
            raise RuntimeError(
                f"Failed to place particle {i}. "
                "Try reducing particle size or number density."
            )

    # Convert lists to arrays
    positions = np.array(positions)
    densities = np.array(accepted_densities)
    radii = np.array(accepted_radii)

    # Combine into final output format
    return np.hstack([positions, densities[:, None], radii[:, None]])


def save_to_csv(data, filename="particles.csv"):
    """
    Save particle data to CSV with the following columns:
    position_0, position_1, ..., position_{dim-1}, density, radius
    """
    dim = data.shape[1] - 2
    headers = [f"position_{i}" for i in range(dim)] + ["density", "radius"]
    np.savetxt(filename, data, delimiter=",", header=",".join(headers), comments="")


# --- Example usage ---
if __name__ == "__main__":
    particles = generate_particles(
        n_particles=1000,
        dim=3,
        box_min=(0, 0, 0),
        box_max=(1000e-6, 1000e-6, 500e-6),
        radius_range=(15e-6, 30e-6),
        density_range=(4386, 4386.1),
        max_attempts=10000,
        seed=42,
    )

    save_to_csv(particles, "feed_stock_particles.csv")

    print("Generated particle configuration and saved to particles.csv")
