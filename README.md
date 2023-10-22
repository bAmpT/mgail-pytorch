## Model-Based Generative Adversarial Imitation Learning (MGAIL)

Pytorch implementation for "End-to-End Differentiable Adversarial Imitation Learning", by Nir Baram, Oron Anschel, Itai Caspi, Shie Mannor.

## Dependencies
Install the required dependencies using the following command:
```bash
pip install -r requirements.txt
```

## References
The code is based on the tensorflow implementation: https://github.com/itaicaspi/mgail

## Running
Run the following command to train the Mujoco Hopper-v2 environment by imitating an expert trained with SAC or PPO:

```python
python main.py
```