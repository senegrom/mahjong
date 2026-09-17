from pathlib import Path
p = Path('neural/modal_app.py')
s = p.read_text()
old = '"policy_inference": {"version": 1, "precision": "bfloat16", "tie_break": "first_policy_index"},'
new = '"policy_inference_rule": {"version": 1, "cuda_46": "bfloat16", "other": "float32", "tie_break": "first_policy_index"},'
assert s.count(old) == 1
p.write_text(s.replace(old, new))
