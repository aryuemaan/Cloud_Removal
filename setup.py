from setuptools import find_packages, setup

with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="liss4-cloud-removal",
    version="1.0.0",
    description="Generative AI-based cloud removal and reconstruction for LISS-IV satellite imagery",
    author="Geospatial AI Team",
    python_requires=">=3.10",
    packages=find_packages(include=["src", "src.*", "api", "api.*"]),
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "liss4-train=src.training.train:main",
            "liss4-infer=src.inference.predict:main",
            "liss4-evaluate=src.evaluation.evaluate:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: GIS",
    ],
)
