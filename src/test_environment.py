import sys
import pandas
import numpy
import sklearn
import requests
from dotenv import load_dotenv
import os

load_dotenv()

print("=" * 50)
print("CodeGuard AI - Environment Test")
print("=" * 50)

print("Python      :", sys.version)
print("Pandas      :", pandas.__version__)
print("NumPy       :", numpy.__version__)
print("Scikit-learn:", sklearn.__version__)
print("Requests    :", requests.__version__)

token = os.getenv("GITHUB_TOKEN")

if token:
    print("GitHub token: FOUND")
else:
    print("GitHub token: NOT FOUND")

print("=" * 50)
print("Environment OK")
print("=" * 50)