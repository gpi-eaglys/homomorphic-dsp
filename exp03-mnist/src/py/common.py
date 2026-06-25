import os

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
EXP_DIR    = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
REPO_DIR   = os.path.abspath(os.path.join(EXP_DIR, ".."))
BLD_DIR    = os.path.join(REPO_DIR, "build")
MNIST_ROOT = os.path.join(REPO_DIR, "assets/mnist")
