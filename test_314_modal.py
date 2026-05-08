import modal
from pathlib import Path

# Match sandbox.py exactly (3.14 alignment)
image = (
    modal.Image.debian_slim(python_version="3.14")
    .pip_install("jax[cpu]")
)

app = modal.App("version-test")

@app.function(image=image, serialized=True)
def test():
    import jax
    print(f"Success! Running JAX {jax.__version__} on Python 3.14")

if __name__ == "__main__":
    with app.run():
        test.remote()
