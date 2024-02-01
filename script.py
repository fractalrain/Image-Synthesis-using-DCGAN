import subprocess
import sys
import os


def create_venv(venv_name):
    try:
        subprocess.run([sys.executable, '-m', 'venv', venv_name], check=True)
    except subprocess.CalledProcessError:
        print(f"Error: Unable to create virtual environment {venv_name}")
        sys.exit(1)


def install_dependencies(requirements_file):
    with open(requirements_file, 'r') as f:
        dependencies = f.readlines()

    for dependency in dependencies:
        dependency = dependency.strip()
        try:
            subprocess.run([os.path.join('venv', 'bin', 'pip'), 'install', dependency], check=True)
        except subprocess.CalledProcessError:
            print(f"Skipping {dependency}. Package not found or cannot be installed.")


if __name__ == "__main__":
    venv_name = 'my_venv'  # Specify the name of the virtual environment
    create_venv(venv_name)

    # Activate the virtual environment for Windows
    if sys.platform == 'win32':
        activate_script = os.path.join(venv_name, 'Scripts', 'activate')
        subprocess.run([activate_script], shell=True)

    requirements_file = 'requirements.txt'
    install_dependencies(requirements_file)
