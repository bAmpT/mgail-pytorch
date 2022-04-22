## Model-Based Generative Adversarial Imitation Learning

Code for ICML 2017 paper "End-to-End Differentiable Adversarial Imitation Learning", by Nir Baram, Oron Anschel, Itai Caspi, Shie Mannor.

## Dependencies
* gym >= 0.8.1
* mujoco-py >= 0.5.7
* torch >= 1.11.0

## Running
Run the following command to train the Mujoco Hopper-v2 environment by imitating an expert trained with TRPO

```python
python main.py
```