DBT-PINN is a novel Physics-Informed Neural Network framework designed for solving coupled differential equation systems.

Unlike traditional PINNs that optimize all equations using a single loss function, DBT-PINN assigns an independent neural network and loss function to each governing equation.

The proposed Dual-Balancing Training (DBT) strategy improves the optimization process through two complementary mechanisms:

- **Inter-balancing:** coordinates the learning progress among coupled equations through alternating optimization.

- **Intra-balancing:** adaptively adjusts the training steps of each sub-network to improve parameter-space exploration.

- ## Features

- Independent PINN architecture for coupled equations
- Dual-balancing training strategy
- Alternating optimization mechanism
- Adaptive training-step adjustment
- Three-degree-of-freedom aircraft simulation example
- Comparison with the conventional PINN method

## Results

Compared with the traditional PINN approach, DBT-PINN achieves:

- Better training stability
- Faster convergence
- Higher prediction accuracy

## Example

The repository includes a three-degree-of-freedom aircraft model for validation.

## Citation

If you find this work useful, please cite:

```bibtex
@article{your_paper,
  title={...},
  author={Zhi Zhu},
  journal={...},
  year={2026}
}
```
