from setuptools import find_packages, setup

setup(
    name="hybrid-multimodal-retrieval",
    version="0.1.0",
    description="Personal multimodal retrieval project based on Flickr30K",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.10",
)
