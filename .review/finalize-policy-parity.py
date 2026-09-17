from pathlib import Path
p = Path('.review/apply-policy-parity.py')
s = p.read_text()
start = s.index("    part = rep(part, '\"audit_share\": audit_share,',")
end = s.index('    return s[:begin]', start)
s = s[:start] + '''    part = rep(part, 'extra_candidates=extra_candidates, audit_share=audit_share,',
               'extra_candidates=extra_candidates, audit_share=audit_share, rollout_batch=rollout_batch,\\n                        policy_inference_rule=dict(version=1, cuda_46="bfloat16", other="float32", tie_break="first_policy_index"),')
''' + s[end:]
exec(compile(s, str(p), 'exec'))
