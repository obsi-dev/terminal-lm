# app/test_pipeline.py
from app.inference import InferenceEngine
from sandbox.executor import run_in_sandbox

engine = InferenceEngine()
print("Loading model...")
engine.load()
print("Loaded.\n")

query = "show current directory"
command = engine.generate(query)
print(f"Query: {query}")
print(f"Generated command: {command}\n")

result = run_in_sandbox(command)
print(f"Sandbox exit code: {result.exit_code}")
print(f"Sandbox stdout: {result.stdout}")
print(f"Sandbox stderr: {result.stderr}")
