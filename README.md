# Diffusion ISIR

--------------------

**Diffusion ISIR** is a novel approach for enhancing diffusion model sampling through **Iterated Sequence Importance Sampling (ISIR)**. By refining the reverse diffusion process, Diffusion ISIR reduces sampling errors and improves the quality of generated samples. It builds on ideas from **Diffusion Rejection Sampling (DiffRS)** and traditional diffusion models like **Elucidated Diffusion Models (EDM)** and **Discriminator-Guided EDM (DG EDM)**.

---

## Repository Structure

- **`isir_code.py`**: Core implementation of Diffusion ISIR.
- **`generate_diffrs.py`**: Script to run DiffRS sampling (based on the DiffRS paper).
- **`classifier_lib.py`**: Contains utility functions, including the time-dependent discriminator for density ratio estimation.
- **`fid_npzs.py`**: Script to compute FID scores for generated samples.
- **`exp.ipynb`**: Notebook with code for running all the methods.

---

## Results

### FID Scores on CIFAR-10
The table below shows the Frechet Inception Distance (FID) scores for four methods tested on CIFAR-10:

| **Method**          | **FID (↓)**  |
|----------------------|--------------|
| EDM                 | 6.81854      |
| DG EDM              | 6.60221      |
| DiffRS              | 6.37657      |
| ISIR Diffusion (Ours) | 6.62911      |

With Diffusion ISIR, we improved results of the basic EDM and are almost on the same level with DG+EDM. But it should be noted, that our results significantly depend on the choice of parameters K (number of ISIR iterations on one step) and N (number of samples considered on each ISIR iteration). These parameters should be tuned, and this requires further research. Probably, this tuning will help to outperform Diffrs.

---

## Future Work

We plan to extend this project by:
- **Parameter Tuning**: Diffusion ISIR's performance depends on the hyperparameters K (iterations per step) and N (samples per iteration). Tuning these values could further enhance the method’s performance, potentially surpassing DiffRS.
- **Extensive Comparisons**: Additional experiments will compare ISIR Diffusion to other state-of-the-art diffusion sampling techniques.


