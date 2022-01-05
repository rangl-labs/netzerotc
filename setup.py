from setuptools import setup, find_packages

setup(
    name="rangl",
    version="0.1.0",
    install_requires=["gym", "matplotlib", "numpy", "pandas", "pycel"],
    packages=find_packages(),
    package_data={
        "rangl": ["compiled_workbook_objects/*.xlsx", "compiled_workbook_objects/*.pkl"]
    },
)
