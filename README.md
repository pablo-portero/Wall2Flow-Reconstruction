# 3D Reconstruction of Turbulence From Wall Data Using a Physics Guided Motion-Transformer Framework

This repository implements a physics-guided deep generative framework for the real-time reconstruction of three-dimensional turbulent channel flows from wall measurements.
The method is motivated by the fact that, in many aerodynamic and industrial applications, only wall-based quantities such as pressure and shear stresses are accessible in real time. The proposed architecture uses these sparse surface measurements to reconstruct the full volumetric velocity field inside the flow domain.
The core idea is to reinterpret the wall-normal direction as a pseudo-temporal axis. In this way, the reconstruction problem is formulated as an image-to-video generation task, where each wall-parallel flow slice is generated sequentially from the wall towards the channel center.
The architecture is composed of different main components:
