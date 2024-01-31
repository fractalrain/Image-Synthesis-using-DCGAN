import subprocess


def install_dependencies(requirements_file):
    with open(requirements_file, 'r') as f:
        dependencies = f.readlines()

    for dependency in dependencies:
        dependency = dependency.strip()
        try:
            subprocess.run(['pip', 'install', dependency], check=True)
        except subprocess.CalledProcessError:
            print(f"Skipping {dependency}. Package not found or cannot be installed.")


if __name__ == "__main__":
    requirements_file = 'requirements.txt'
    install_dependencies(requirements_file)
