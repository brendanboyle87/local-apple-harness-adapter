"""Fetch the pinned SmolDataEnvs tasks and build offline ARM64 images."""

import argparse
import hashlib
import json
import shutil
import subprocess
import tomllib
from functools import cache
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

HERE = Path(__file__).resolve().parent
DATASET = "FineEnvs/SmolDataEnvs-harbor-train"
REVISION = "588744be728257abfb8d4984ec8fab2acbd1f7e1"
FIXTURES = {"pi": "write-answer", "opencode": "write-answer-opencode"}
ASSETS = ("instruction.md", "task.toml", "tests/test.sh", "tests/grader.py")
API = HfApi(token=False)
INSTALL = "pip install --quiet math-verify >/dev/null 2>&1 || true\n"


def docker(*arguments, **options):
    return subprocess.run(["docker", *map(str, arguments)], check=True, text=True, **options)


def file_digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def matches(path, entry):
    return (
        path.is_file()
        and path.stat().st_size == entry["bytes"]
        and file_digest(path) == entry["sha256"]
    )


@cache
def bucket_files(bucket, prefix):
    return {
        Path(file.path).name: file
        for file in API.list_bucket_tree(bucket, prefix=prefix, recursive=True, token=False)
        if file.type == "file" and file.path.startswith(f"{prefix}/")
    }


def stage_inputs(task, destination, storage):
    destination.mkdir(parents=True, exist_ok=True)
    storage.mkdir(parents=True, exist_ok=True)
    for entry in task["inputs"]:
        name = entry["filename"]
        cached = storage / entry["sha256"]
        if not matches(cached, entry):
            source = bucket_files(task["bucket"], task["prefix"])[name]
            temporary = cached.with_suffix(".download")
            API.download_bucket_files(
                task["bucket"], [(source, temporary)], raise_on_missing_files=True, token=False
            )
            if not matches(temporary, entry):
                raise ValueError(f"Input contents changed for {task['name']}/{name}")
            temporary.replace(cached)
        if not matches(destination / name, entry):
            shutil.copyfile(cached, destination / name)
    if {path.name for path in destination.iterdir()} != {
        entry["filename"] for entry in task["inputs"]
    }:
        raise ValueError(
            f"Unexpected files in {destination}. Stage the tasks in a fresh output directory."
        )


def build_base(harness):
    fixture = HERE.parent / "harbor" / FIXTURES[harness] / "environment"
    client = f"local-apple-harness-{harness}:fixture"
    base = f"local-apple-harness-smoldataenvs-{harness}:base"
    docker("build", "--platform=linux/arm64", "-t", client, fixture)
    docker(
        "build",
        "--platform=linux/arm64",
        "--build-arg",
        f"HARNESS_IMAGE={client}",
        "-t",
        base,
        HERE,
    )
    return base


def stage_task(task, root, base, harness):
    target = root / "tasks" / task["name"]
    (target / "tests").mkdir(parents=True, exist_ok=True)
    files = {
        asset: Path(
            hf_hub_download(
                DATASET,
                f"{task['path']}/{asset}",
                repo_type="dataset",
                revision=REVISION,
                token=False,
            )
        )
        for asset in ASSETS
    }
    upstream = tomllib.loads(files["task.toml"].read_text())
    verifier_env = upstream["verifier"]["env"]
    shutil.copyfile(files["instruction.md"], target / "instruction.md")
    shutil.copyfile(files["tests/grader.py"], target / "tests/grader.py")
    verifier = target / "tests/test.sh"
    verifier.write_text(files["tests/test.sh"].read_text().replace(INSTALL, ""))
    verifier.chmod(0o755)

    inputs = target / "environment/input"
    stage_inputs(task, inputs, root / "inputs")
    tag = f"local-apple-harness-smoldataenvs-{harness}:{task['name']}"
    docker(
        "build",
        "--platform=linux/arm64",
        "--tag",
        tag,
        "--file",
        "-",
        inputs,
        input=f"FROM {base}\nCOPY . /home/user/input/\n",
    )
    config = (
        'version = "1.0"\n\n[agent]\nuser = "node"\ntimeout_sec = 150.0\n'
        '[verifier]\nuser = "root"\ntimeout_sec = 30.0\n'
        '[environment]\nnetwork_mode = "no-network"\ncpus = 2\nmemory_mb = 2048\n'
        f'docker_image = "{tag}"\n\n[verifier.env]\n'
    )
    config += "".join(
        f"{json.dumps(key)} = {json.dumps(value, ensure_ascii=False)}\n"
        for key, value in verifier_env.items()
    )
    if tomllib.loads(config)["verifier"]["env"] != verifier_env:
        raise ValueError(f"Could not preserve verifier configuration for {task['name']}")
    env_flags = [
        part for key, value in verifier_env.items() for part in ("--env", f"{key}={value}")
    ]
    test = docker(
        "run",
        "--rm",
        "-i",
        "--pull=never",
        "--network=none",
        *env_flags,
        "-v",
        f"{target / 'tests'}:/tests:ro",
        tag,
        "python",
        "/tests/grader.py",
        input=f"{verifier_env['EXPECTED_ANSWER']}\n",
        stdout=subprocess.PIPE,
    )
    if float(test.stdout.strip()) != 1:
        raise ValueError(f"The upstream reference answer failed its verifier for {task['name']}")
    (target / "task.toml").write_text(config)
    return {
        "name": task["name"],
        "source": task["prefix"],
        "difficulty": task["difficulty"],
        "path": str(target.relative_to(root)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=("warmup", "train", "heldout"))
    parser.add_argument("--harness", choices=FIXTURES, default="pi")
    parser.add_argument("--limit", type=int, help="stage only the first N tasks")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    tasks = [json.loads(line) for line in (HERE / f"{args.split}.jsonl").read_text().splitlines()]
    base = build_base(args.harness)
    records = [stage_task(task, root, base, args.harness) for task in tasks[: args.limit]]
    manifest = {
        "dataset": DATASET,
        "revision": REVISION,
        "split": args.split,
        "harness": args.harness,
        "tasks": records,
    }
    temporary = root / "manifest.json.pending"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(root / "manifest.json")
    print(f"Prepared {len(records)} {args.harness} tasks in {root}")


if __name__ == "__main__":
    main()
