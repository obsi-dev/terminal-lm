from inference import InferenceEngine

engine = InferenceEngine()

print("Loading model...")
engine.load()
print("Loaded. Testing...")

test_prompts = [
    "show current directory",
    "list all running processes",
    "compress the file report.txt using bzip2",
]

for prompt in test_prompts:
    result = engine.generate(prompt)
    print(f"IN: {prompt}")
    print(f"OUT: {result}")
    print()
