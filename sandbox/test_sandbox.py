import os
from executor import run_in_sandbox

result = run_in_sandbox("find /home/sandboxuser -maxdepth 2", working_dir=os.getcwd())
print(result.stdout)
