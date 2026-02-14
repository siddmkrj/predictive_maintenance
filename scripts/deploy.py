import os
from huggingface_hub import HfApi, create_repo

SPACE_REPO_ID = "mukherjee78/predictive-maintenance-app"
SPACE_SDK = "docker"

api = HfApi()

print(f"Creating Hugging Face Space: {SPACE_REPO_ID}")
create_repo(
    repo_id=SPACE_REPO_ID,
    repo_type="space",
    space_sdk=SPACE_SDK,
    private=False,
    exist_ok=True
)
print("Space created successfully.")

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

deployment_files = {
    os.path.join(project_root, "app.py"): "app.py",
    os.path.join(project_root, "requirements.txt"): "requirements.txt",
    os.path.join(project_root, "Dockerfile"): "Dockerfile",
}

for local_path, repo_path in deployment_files.items():
    print(f"Uploading {local_path} → {repo_path}")
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=repo_path,
        repo_id=SPACE_REPO_ID,
        repo_type="space"
    )

print(f"\nDeployment complete!")
print(f"App URL: https://huggingface.co/spaces/{SPACE_REPO_ID}")
