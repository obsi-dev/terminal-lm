# sandbox/test_sandbox.py
from executor import run_in_sandbox

result = run_in_sandbox("echo hello && ls -la")
print("Exit code:", result.exit_code)
print("Stdout:", result.stdout)
print("Stderr:", result.stderr)
