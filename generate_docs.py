import os
import subprocess

# Paths
src_root = os.path.join("src", "kinematics_library")
output_dir = "docs"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Set PYTHONPATH for subprocess
env = os.environ.copy()
env["PYTHONPATH"] = os.path.abspath("src")

# Walk through files in src/kinematics_library
for root, dirs, files in os.walk(src_root):
    for file in files:
        if file.endswith(".py") and not file.startswith("__"):
            rel_path = os.path.relpath(os.path.join(root, file), "src")
            module_name = rel_path.replace(os.sep, ".").replace(".py", "")
            print(f"Generating docs for {module_name}...")

            try:
                subprocess.run(
                    ["python", "-m", "pydoc", "-w", module_name],
                    check=True,
                    cwd=os.getcwd(),  # run from project root,
                    env=env,
                )
            except subprocess.CalledProcessError as e:
                print(f"Failed to generate docs for {module_name}: {e}")
