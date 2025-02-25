.. DACTI documentation master file, created by
   sphinx-quickstart on Thu Feb 20 13:45:33 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to DACTI's documentation!
=================================

DACTI (Dynamic Adaptive Conservative Time Integration) is a space-time adaptive code that solves conservation laws, e.g., Navier-Stokes-Fourier system for compressible flows. The code is based on an explicit finite volume method. Tree-based adaptive mesh refinement (AMR) is used for spatial discretization, which is further combined with an adaptive conservative time integration scheme to localize the most severe time step constraints. 

.. toctree::
   :maxdepth: 2
   :caption: Contents

   getting_started

.. toctree:: 
   :caption: References

