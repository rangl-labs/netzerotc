from setuptools import setup

setup(
    # name="reference_environment",
    name="reference_environment_direct_deployment",
    version="0.1.0",
    install_requires=["gym", "matplotlib", "numpy", "pandas", "pycel"],
)

setup(
    name="open_loop_env",
    version="0.1.0",
    install_requires=["gym", "matplotlib", "numpy", "pandas", "pycel"],
)

setup(
    name="closed_loop_env",
    version="0.1.0",
    install_requires=["gym", "matplotlib", "numpy", "pandas", "pycel"],
)