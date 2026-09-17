from pathlib import Path
p = Path('.review/apply-policy-parity.py')
s = p.read_text()
start = s.index("    part = rep(part, '\"audit_share\": audit_share,',")
end = s.index('    return s[:begin]', start)
old = 'extra_candidates=extra_candidates, audit_share=audit_share,\n                        placement_head_sha256='
new = ('extra_candidates=extra_candidates, audit_share=audit_share, rollout_batch=rollout_batch,\n'
       '                        policy_inference_rule=dict(version=1, cuda_46="bfloat16", other="float32", tie_break="first_policy_index"),\n'
       '                        placement_head_sha256=')
replacement = '    part = rep(part, ' + repr(old) + ',\n               ' + repr(new) + ')\n'
s = s[:start] + replacement + s[end:]
# The runner may publish source, but its token may not edit CI workflows.
# CI changes are handled separately through the authorized repository connector.
s = s.replace("edit('.github/workflows/neural.yml', ci)", "# CI file is not a runner-published source file")
exec(compile(s, str(p), 'exec'))
p = Path('neural/searched.py')
s = p.read_text()
old = 'def play_lookahead(net, arena, *, device="cuda", temperature=0.0, passes=8000):'
assert s.count(old) == 1
s = s.replace(old, 'def play_lookahead(net, arena, *, device="cuda", temperature=0.0, passes=8000,\n                   step=policy_inference.DEFAULT_ROLLOUT_BATCH):')
assert s.count('        step = 8192\n') == 1
s = s.replace('        step = 8192\n', '')
start = s.index('def play_lookahead(')
at = s.index('    require_native_search(net)\n', start)
s = s[:at] + '    policy_inference.validate_batch(step)\n' + s[at:]
p.write_text(s)
