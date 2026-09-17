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
exec(compile(s, str(p), 'exec'))
