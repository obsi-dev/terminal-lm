import docker
import os
import tempfile
import shutil
from dataclasses import dataclass

client = docker.from_env()


def _ignore_hidden(directory: str, contents: list[str]) -> list[str]:
    return [name for name in contents if name.startswith(".")]


@dataclass
class SandboxResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool


def run_in_sandbox(
    command: str,
    working_dir: str | None = None,
    timeout: int = 10,
) -> SandboxResult:
    volumes = {}
    temp_copy = None

    if working_dir:
        temp_copy = tempfile.mkdtemp(prefix="terminal-lm_sandbox_")
        shutil.copytree(
            working_dir, temp_copy, dirs_exist_ok=True, ignore=_ignore_hidden
        )

        for root, dirs, files in os.walk(temp_copy):
            for d in dirs:
                os.chmod(os.path.join(root, d), 0o777)
            for f in files:
                os.chmod(os.path.join(root, f), 0o666)

        volumes[temp_copy] = {"bind": "/home/sandboxuser", "mode": "rw"}

    container = None

    try:
        container = client.containers.run(
            image="terminal-lm",
            command=["bash", "-c", command],
            volumes=volumes,
            network_disabled=True,
            mem_limit="256m",
            nano_cpus=1_000_000_000,
            pids_limit=64,
            detach=True,
            remove=False,
        )
        try:
            result = container.wait(timeout=timeout)
            timed_out = False
        except Exception:
            container.kill()
            result = {"StatusCode": -1}
            timed_out = True

        stdout = container.logs(stdout=True, stderr=False).decode(errors="replace")
        stderr = container.logs(stdout=False, stderr=True).decode(errors="replace")

        return SandboxResult(
            exit_code=result["StatusCode"],
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
        )
    finally:
        if container:
            container.remove(force=True)
        if temp_copy:
            shutil.rmtree(temp_copy, ignore_errors=True)
