# Install

```bash
git clone https://github.com/RobMina/scint-fibre-fastsim.git
cd scint-fibre-fastsim
python -m pip install -e '.[test]'
python -m pytest -q
```

Optional `.[jax]` is enough to import the transport modules; `.[test]` adds
pytest. CI uses the JAX CPU backend only. For NVIDIA / AMD / Intel devices see
[hardware](../explanation/hardware.md).
