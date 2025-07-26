# Pyseter

A Python package that sorts images by an automatically generated ID before photo-identification. 

## Installation

### New to Python?

While most biologists use R, Python is the language for machine learning tasks with images. If you're new to Python, please follow these steps to installing Python and conda. Conda is an important tool for managing packages in Python. Packages in R are handled behind the scenes. Python, however, requires a more hands on approach.

   - Download [Python](https://www.python.org/downloads/) (Windows users only)
   - Download and install [Miniforge](https://conda-forge.org/download/) 

#### Create a new environment

Then, you'll need to interact with the dreaded command line, which will depend on your operating system. Are you on Windows? Open the "miniforge prompt" in your start menu. Are you on Mac? Open the Terminal application. Then, you'll create the environment that the package will live in. Environments are walled off areas where we can install packages, which helps prevent conflicts between packages.

```
conda create -n pyseter_env
conda activate pyseter_env
```

Your environment is ready to go! Try installing your first package, pip. Pip is another way of installing Python packages, and will be helpful for installing PyTorch and pyseter (see below).

```
conda install pip -y
```

#### Interacting with Python

There are several different ways to interact with Python. The most common way is through a *Jupyter Notebook*, which is similar to an RMarkdown document or a Quarto document. There are several ways to open Jupyter Notebooks. 

The most common way is to [install Jupyter](https://jupyter.org/install) via the command line. Below, I also install the [ipykernel](https://ipython.readthedocs.io/en/stable/install/kernel_install.html#kernels-for-different-environments), which helps you select the right conda environment in [Jupyter Lab](https://docs.jupyter.org/en/latest/).

```
conda activate pyseter_env
conda install jupyter ipykernel -y
python -m ipykernel install --user --name pyseter --display-name "Python (pyseter)"
```

Then you can open Jupyter Lab (similar to R Studio) with 

```
jupyter lab
```

Alternatively, you can open Jupyter Notebooks with [VS Code](https://code.visualstudio.com/download) (my favorite), or [Positron](https://positron.posit.co) (a hybrid between R Studio and VS Code).

### Install PyTorch with GPU support

Installing PyTorch will allow users to extract features from images, i.e., identify individuals in images. **This will only be realistic for users with a CUDA compatible GPU.**. Without a GPU, extracting image features will be painfully slow (hours if not days), compared with minutes. 

PyTorch can be a little finicky. I recommend following [these instructions](https://pytorch.org/get-started/locally/). For Windows users, the command might be

```
conda activate pyseter
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```
although this will depend on your version of CUDA. If you don't know, try the command above. PyTorch is pretty big (over a gigabyte), so this may take a few minutes.

### Install pyseter

Finally, we can install pyseter. If you haven't already, activate your py

```
conda activate pyseter_env
pip3 install pyseter
```

And that's it!

## Getting started

Open a Jupyter Notebook. You can do this through Jupyter Lab, VS Code, or Positron. The first step is verifying your Python installation
