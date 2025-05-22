import sys
import os
import subprocess
import yaml
import toml
import site
import re


def normalize_pkg_name(pkg):
    # Elimina operadores de versión y espacios: "pyqubo>=0.5" -> "pyqubo"
    return re.split(r"[<>=!~]+", pkg.strip())[0].lower()


def get_conda_installed_packages(env_name=None):
    cmd = ["conda", "list"]
    if env_name:
        cmd.extend(["-n", env_name])
    result = subprocess.run(cmd, capture_output=True, text=True)
    packages = set()
    for line in result.stdout.splitlines():
        if not line.startswith("#") and line.strip():
            name = line.split()[0].lower()
            packages.add(name)
    return packages


def get_env_yaml_packages(path="environment.yml"):
    with open(path, "r") as f:
        env_data = yaml.safe_load(f)

    pkgs = set()
    for item in env_data.get("dependencies", []):
        if isinstance(item, str):
            if item.lower() != "pip":
                pkgs.add(normalize_pkg_name(item))
        elif isinstance(item, dict) and "pip" in item:
            for pip_pkg in item["pip"]:
                pkgs.add(normalize_pkg_name(pip_pkg))
    return pkgs


def get_pyproject_packages(path="pyproject.toml"):
    with open(path, "r") as f:
        data = toml.load(f)

    pkgs = set()
    deps = data.get("project", {}).get("dependencies", [])
    for dep in deps:
        pkgs.add(normalize_pkg_name(dep))
    return pkgs


if __name__ == "__main__":
    print("=" * 60)
    print("🐍 Intérprete Python activo:")
    print(sys.executable)

    print("\n📦 sys.prefix (ruta entorno):")
    print(sys.prefix)

    conda_prefix = os.environ.get("CONDA_PREFIX", "No detectado")
    print("\n🌐 CONDA_PREFIX:")
    print(conda_prefix)

    print("\n📚 Rutas site-packages:")
    for path in site.getsitepackages():
        print(" -", path)
    print("👤 Usuario site-packages:", site.getusersitepackages())

    env_yaml = get_env_yaml_packages()
    pyproject = get_pyproject_packages()

    print("\n📄 Paquetes definidos en environment.yml:")
    for pkg in sorted(env_yaml):
        print(" -", pkg)

    print("\n📄 Paquetes definidos en pyproject.toml:")
    for pkg in sorted(pyproject):
        print(" -", pkg)

    # Obtener nombre del entorno desde CONDA_PREFIX (último segmento del path)
    env_name = (
        os.path.basename(conda_prefix) if conda_prefix != "No detectado" else None
    )
    conda_installed = get_conda_installed_packages(env_name)

    combined_required = env_yaml.union(pyproject)
    missing = combined_required - conda_installed

    print("\n🔍 Faltan estos paquetes según los archivos de definición:")
    if missing:
        for pkg in sorted(missing):
            print(" ❌", pkg)
    else:
        print(" ✅ Ninguno")

    print("\n✅ Verificación finalizada.")
    print("=" * 60)
