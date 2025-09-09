# Summary

A complete image dehazing pipeline implementing the Dark Channel Prior (He et al.) with Guided Image Filtering for edge-preserving refinement, plus optional experiments toward a Multi-Scale CNN dehazing approach (Ren et al.). The repository estimates transmission, atmospheric light, and restores the haze-free radiance, with tunable parameters and reproducible examples.

# Features
Dark Channel Prior implementation for single-image dehazing with patch-wise minima.
Atmospheric light estimation with two strategies: He et al. top-percentile approach and an alternative to reduce overestimation on sky-less scenes.
Transmission map estimation with ω control and t0 floor to preserve natural depth.
Guided filter refinement of the transmission using the original image as guide for crisp edges.
Parameter studies for Ω, ε, r, ω, with qualitative comparisons on example images.
