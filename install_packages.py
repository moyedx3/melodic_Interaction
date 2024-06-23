packages = [
    "librosa",
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    "scipy",
    "itertools",
    "sklearn"  # The correct package name for sklearn is scikit-learn
]

for package in packages:
    try:
        __import__(package)
        print(f"{package} is installed correctly.")
    except ImportError as e:
        print(f"Error: {package} is not installed. {e}")
    except Exception as e:
        print(f"Error: An issue occurred while checking {package}. {e}")

print("Package check completed.")
