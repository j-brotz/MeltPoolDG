# Wall Wetting 

The cases discussed here simulate the ones studied in the *Model problem* section of *A conservative level set method for contact line dynamics* by Zahedi *et al.* (2009) [^1]

## Feature

- `mp-reinit` with wetting boundary condition

## Folder Structure

Since many different cases are used for the studies, a multiple folder structure is used. 

```
wall_wetting_zahedi_2009_comparison/
├── doc
├── meltpooldg_results
│   ├── logs
│   ├── zahedi_wall_wetting_000
│   ├── zahedi_wall_wetting_001
│   ├── ...
│   └── zahedi_wall_wetting_020
├── parameter_files
├── scripts
│   ├── post_processing
│   └── pre_processing
├── tests
└── zahedi_results
    ├── fig2
    ├── fig3
    ├── fig4
    ├── fig5
    │   ├── fig5a
    │   └── fig5b
    └── fig6

```
*Note that some of the folders listed above are only generated after running and post-processing the cases.* 

A brief overview of the folder structure is presented here:

- Folder that contains figures for this Markdown file: `doc/`
- Folder that contains simulation results from MeltPoolDG: `meltpooldg_results/`
- Folder that contains JSON files for the simulated cases: `parameters_files/`
- Folder that contains scripts for pre-processing and post-processing: `scripts/`
- Folder that contains code testing files: `tests/`
- Folder that contains simulation results from Zahedi: `zahedi_results/`

To learn more on how to generate, launch and post-process the cases, please refer to the [README.md](./scripts/README.md) file located in the `scripts/` folder.

## Problem Description

The initial configuration of the problem is given by the figure 1.

<figure align="center">
  <img src="./doc/initial_state.png" style="border: 2px solid black" width="800"/>
</figure> 

**Figure 1**. Initial state of the problem. 

In a 2D domain $\Omega = [0,2] \times [0,2]$, we initiate the initial level set field $\phi(\vec{x})$ with:

$$\phi(x) = -\tanh\left(\frac{d(x)}{2\varepsilon} \right)$$

where
- $d(x) = x_\text{interface} - x$ is the signed distance from the interface 
- $\varepsilon = f_\varepsilon h$ is the interface thickness parameter with $f_\varepsilon$ a user-inputted constant factor and $h$ the cell-size. For all cases, $h$ is fixed to $0.005$. 

We set the interface to be a vertical line at $x=0 \,(\Rightarrow x_\text{interface} = 0)$.

The reinitialization equation takes the following form:

$$ \frac{\partial\phi}{\partial \tau} + \nabla \cdot \left[\phi (1 - \phi) \vec{n}\right] - \nabla \cdot \left[\varepsilon_\text{n} (\nabla \phi \cdot \vec{n} ) \vec{n} \right] - \nabla \cdot \left[\varepsilon_\text{t} (\nabla \phi \cdot \vec{t} ) \vec{t} \right] = 0$$

where
- $\tau$ is the independent time variable
- $\vec{n}$ is the unit normal vector to the interface pointing in the direction $\nabla \phi$. Note that $\vec{n}$ is only computed once before starting the reinitialization process and the same vector $\vec{n}$ is used when solving the transient problem
- $\varepsilon_\text{n}$ is the diffusion coefficient in the normal direction to the interface. For all simulation cases, this parameter is set as $\varepsilon_\text{n} = \varepsilon$
- $\varepsilon_\text{t}$ is the diffusion coefficient in the tangential direction to the interface
- $\vec{t}$ is the unit tangential vector to the interface pointing in the direction orthogonal to $\vec{n}$

As defined in [^1], the time-step is given by $\Delta \tau = \frac{h^2}{2(\varepsilon_\text{n} + \varepsilon_\text{t})}$

To ensure continuity of the solution, $\vec{n}$ is filtered and computed by solving the following linear equation:

$$ \vec{n} - \nabla \cdot (\eta_\text{n} h^2 \nabla \vec{n}) = \frac{\nabla \phi}{\lVert \nabla \phi\rVert} $$

where $\eta_\text{n} = (f_\gamma f_\varepsilon)^2$ is the filtering factor with $f_\gamma$ the factor used in [^1] to do a parametric sweep on the regularization parameter $(\gamma = f_\gamma \varepsilon_\text{n})$ they define.

In order to impose a given contact angle at the bottom wall, an interface normal vector Dirichlet boundary condition is imposed at the wall.

As we are interested in the bottom wall, for a given static contact angle $(\alpha_\text{s})$ we have:

$n_x|_{y=0} = \sin(\alpha_\text{s})$

$n_y|_{y=0} = \cos(\alpha_\text{s})$

Both applications use second order accurate spatial schemes:
- Zahedi *et al.* [^1] uses a conservative 2nd order finite difference scheme, and;
- MeltPoolDG uses continuous, linear piecewise ($Q_1$) finite elements.

Furthermore, the applications have different temporal schemes:
- Zahedi *et al.* [^1] uses a 2nd order Runge-Kutta scheme, and;
- MeltPoolDG uses a 1st order semi-implicit BDF scheme.

##  Studied cases

Two different studies are conducted here to compare the implementation with the one of Zahedi *et al.* (2009) [^1]. The studies go as follows:

1. **Study 1: The influence of $\eta_\text{n}$ on the contact angle.**

   While fixing $\alpha_\text{s} = 45^\circ$, $\varepsilon_\text{n} = 8h$ and $\varepsilon_\text{t} = 6\varepsilon_\text{n}$, a parametric sweep is conducted on $\eta_\text{n}$ with $f_\gamma \in \{1.0, 2.5, 5.0, 10.0\}$. 

2. **Study 2: The influence of the ratio $\frac{\varepsilon_\text{t}}{\varepsilon_\text{n}}$ on the contact angle.**
   1. First, we set $\alpha_\text{s} = 45^\circ$, $\varepsilon_\text{n} = 8h$ and $f_\gamma = 2.5$. We then run a parametric sweep for $\frac{\varepsilon_\text{t}}{\varepsilon_\text{n}} \in \{0.5, 1, 3, 12, 24, 48, 96, 192\}$.
   2. Second, keeping the same $\varepsilon_\text{n}$ and $f_\gamma$, we set $\alpha_\text{s} = 25^\circ$ and run a parametric sweep for $\frac{\varepsilon_\text{t}}{\varepsilon_\text{n}} \in \{0.5, 1, 3, 6, 12, 24, 48, 96, 192\}$.

In both studies, the simulation end time ($\tau_\text{end}$) is determined with the steady-state criterion $\frac{\lVert \phi^\text{n+1} - \phi^\text{n} \rVert}{\lVert \phi^\text{n+1} \rVert} \lt 10^{-5}$ as it is done in  [^1].

## Simulation Results

### Study 1

Figure 2 displays the comparison between the interface shape for the different $\eta_\text{n}$ values. As also observed by Zahedi *et al.* [^1], an increasing filtering factor leads to a larger region affected by the contact point. Conversely, using smaller $\eta_\text{n}$ leads to a smaller region affected by the contact point, but larger curvature variations are observed near the contact point. As mentioned in [^1], this indicates that the mesh will have to be fine enough to correctly capture the curvature. 

It is also interesting to note that for higher values of $\eta_\text{n}$, MeltPoolDG's solutions show more lateral displacement of the interface at the wall, but approaches better the prescribed $\alpha_\text{s} = 45^\circ$ as reported in table 1.

<figure align="center">
  <img src="./doc/zahedi_comparison_figure2.png"  width="800"/>
</figure> 

**Figure 2**. Comparison of the isocontour at $\tau_\text{end}$ of the interface $(\phi=0)$ for different values of $\eta_\text{n}$. Here, $\alpha_\text{s} = 45^\circ$, $\varepsilon_\text{n} = 8h$, and $\varepsilon_\text{t} = 6 \varepsilon_\text{n}$.

In figure 3, a close up of figure 2 near the contact point region is shown. It can be observed that for the higher values of $\eta_\text{n}$, the unit normal vectors, indicated by the arrows, transition more gradually, which results in the larger transition region mentioned earlier. 

<figure align="center">
  <img src="./doc/zahedi_comparison_figure2_normal_vector_closeup.png"  width="800"/>
</figure> 

**Figure 3**. Close up near the contact point of the comparison of the isocontour at $\tau_\text{end}$ of the interface $(\phi=0)$ for different values of $\eta_\text{n}$. Here, $\alpha_\text{s} = 45^\circ$, $\varepsilon_\text{n} = 8h$, and $\varepsilon_\text{t} = 6 \varepsilon_\text{n}$. The arrows represent unit normal vectors to the interface.

Figure 4 shows the comparison of time evolution of the computed contact angle. Note that, in figure 4, some curves ($\eta_\text{n} = 1600$ and $\eta_\text{n} = 6400$) from MeltPoolDG's simulations have not "plateaued" yet. However, for the lower $\eta_\text{n}$ values, we observe that MeltPoolDG converges towards steady-state at a higher rate.

<figure align="center">
  <img src="./doc/zahedi_comparison_figure3.png"  width="800"/>
</figure> 

**Figure 4**. Comparison of the time evolution of the contact angle for different $\eta_\text{n}$. Here, $\alpha_\text{s} = 45^\circ$, $\varepsilon_\text{n} = 8h$, and $\varepsilon_\text{t} = 6 \varepsilon_\text{n}$.

**Table 1**: Comparison of the final contact angle for different $\eta_\text{n}$ values with $\alpha_\text{s} = 45^\circ$, $\varepsilon_\text{n} = 8h$, and $\varepsilon_\text{t} = 6 \varepsilon_\text{n}$. 


| $\eta_\text{n}$ | $f_\gamma$ | $\alpha_\text{Zahedi}$ $[^\circ]$ | $\alpha_\text{MeltPoolDG}$ $[^\circ]$ |
|-----------------|------------|-----------------------------------|---------------------------------------|
| 64              | 1          | 47.5                              | 48.5                                  |
| 400             | 2.5        | 47.4                              | 47.4                                  |
| 1600            | 5          | 47.2                              | 46.6                                  |
| 6400            | 10         | 46.7                              | 45.7                                  |


### Study 2

Figure 5 compares the interface position at the end of the simulations for different values of $\varepsilon_\text{t}$ such that $\frac{\varepsilon_\text{t}}{\varepsilon_\text{n}} \in \{3, 12, 48, 96, 192\}$ when $\alpha_\text{s} = 45^\circ$. Note that, the curve $\frac{\varepsilon_\text{t}}{\varepsilon_\text{n}} = 96$ was added here even though [^1] don't report the results. Even though MeltPoolDG's results do not match [^1], a similar evolution in function of $\frac{\varepsilon_\text{t}}{\varepsilon_\text{n}}$ is observed. Indeed, alike [^1], as $\frac{\varepsilon_\text{t}}{\varepsilon_\text{n}}$ increases, the interface converges towards a stable solution. A close up of this evolution is shown in figure 6. 

<figure align="center">
  <img src="./doc/zahedi_comparison_figure5a.png" width="800"/>
</figure> 

**Figure 5**. Comparison of the isocontour at $\tau_\text{end}$ of the interface $(\phi=0)$ for different values of $\varepsilon_\text{t}$. Here, $\alpha_\text{s} = 45^\circ$, $\varepsilon_\text{n} = 8h$, and $\eta_\text{n} = 400$.

<figure align="center">
  <img src="./doc/zahedi_comparison_figure5b_normal_vector.png" width="800"/>
</figure> 

**Figure 6**. Close up near the contact point of the comparison of the isocontour at $\tau_\text{end}$ of the interface $(\phi=0)$ for different values of $\varepsilon_\text{t}$. Here, $\alpha_\text{s} = 45^\circ$, $\varepsilon_\text{n} = 8h$, and $\eta_\text{n} = 400$. The arrows represent unit normal vectors to the interface.


Figures 7 and 8, analogous to figures 5 and 6, show the interface position for different values of $\varepsilon_\text{t}$ when $\alpha_\text{s} = 25^\circ$. Similar tendencies are observed here, indicating that the difference observed with the results of Zahedi *et al.* [^1] is not related to the imposed static angle.

<figure align="center">
  <img src="./doc/zahedi_figure5a_25deg.png" width="800"/>
</figure> 

**Figure 7**. Comparison of the isocontour at $\tau_\text{end}$ of the interface $(\phi=0)$ for different values of $\varepsilon_\text{t}$. Here, $\alpha_\text{s} = 25^\circ$, $\varepsilon_\text{n} = 8h$, and $\eta_\text{n} = 400$.

<figure align="center">
  <img src="./doc/zahedi_figure5b_25deg_normal_vector.png" width="800"/>
</figure> 

**Figure 8**. Close up near the contact point of the comparison of the isocontour at $\tau_\text{end}$ of the interface $(\phi=0)$ for different values of $\varepsilon_\text{t}$. Here, $\alpha_\text{s} = 25^\circ$, $\varepsilon_\text{n} = 8h$, and $\eta_\text{n} = 400$. The arrows represent unit normal vectors to the interface.


Figure 9 shows the computed contact angle evolution for different values of $\varepsilon_\text{t}$ when $\alpha_\text{s} = 45^\circ$. As observed in the first study, MeltPoolDG converges to its final contact angle value at a higher rate. Additionally, on figure 9, results of Zahedi *et el.* [^1]  with lower $\varepsilon_\text{t}$ have not reached steady-state yet. However, in figure 10, it can be observed that for lower $\frac{\varepsilon_\text{t}}{\varepsilon_\text{n}}$ values and $\alpha_\text{s} = 25^\circ$ the results of MeltPoolDG are in fact better.

<figure align="center">
  <img src="./doc/zahedi_comparison_figure4.png" width="800"/>
</figure> 

**Figure 9**. Comparison of the time evolution of the contact angle for different $\varepsilon_\text{t}$. Here, $\alpha_\text{s} = 45^\circ$, $\varepsilon_\text{n} = 8h$, and $\eta_\text{n} = 2.5$.

<figure align="center">
  <img src="./doc/zahedi_comparison_figure6.png"  width="800"/>
</figure> 

**Figure 10**. Comparison of the contact angle ratio $\left(\frac{\alpha}{\alpha_\text{s}}\right)$ in function of the diffusion coefficient ratio $\left(\frac{\varepsilon_\text{t}}{\varepsilon_\text{n}}\right)$ for $\alpha_\text{s} \in \{ 25, 45\}$ with $\varepsilon_\text{n} = 8h$, and $\eta_\text{n} = 400$.

## Reference

[^1]: S. Zahedi, K. Gustavsson, and G. Kreiss, “A conservative level set method for contact line dynamics,” *J. Comput. Phys.*, vol. 228, no. 17, pp. 6361–6375, Sep. 2009, doi: 10.1016/j.jcp.2009.05.043.
