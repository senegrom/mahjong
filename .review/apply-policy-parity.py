"""One-time guarded source edit; removed before publication to main."""
from pathlib import Path
import ast

changed = []
def edit(path, transform):
    p = Path(path)
    old = p.read_text()
    new = transform(old)
    assert new != old, f'No change: {path}'
    if path.endswith('.py'):
        ast.parse(new)
    p.write_text(new)
    changed.append(path)

def rep(s, old, new, count=1):
    assert s.count(old) == count, (old[:120], s.count(old), count)
    return s.replace(old, new)

def contract(s):
    s = rep(s, 'import torch\n', 'import torch\n\nfrom . import policy_inference\n')
    s = rep(s, 'return torch.argsort(logits, dim=1, descending=True).cpu().numpy()',
            'return policy_inference.order(logits).cpu().numpy()', 2)
    s = rep(s, 'torch.argsort(after_logits[:, :POSITIONS].float(), dim=1, descending=True)',
            'policy_inference.order(after_logits[:, :POSITIONS])')
    s = rep(s, 'logits, value, guessed = net.everything(root_planes, root_mask)',
            'logits, value, guessed = policy_inference.everything(net, root_planes, root_mask)')
    s = rep(s, 'after_logits, _after_value, _after_hands = net.everything(after_planes, after_mask)',
            'after_logits, _after_value, _after_hands = policy_inference.everything(net, after_planes, after_mask)')
    s = rep(s, '    fast = getattr(net, "policy_only", None)\n    return fast(planes, legal) if callable(fast) else net(planes, legal)[0]',
            '    return policy_inference.policy_logits(net, planes, legal)')
    s = rep(s, 'def play_lookahead(self, arena, device="cuda", temperature=0.0, passes=8000) -> int:',
            'def play_lookahead(self, arena, device="cuda", temperature=0.0, passes=8000,\n                       batch_size=policy_inference.DEFAULT_ROLLOUT_BATCH) -> int:')
    s = rep(s, '        net = self.net\n        follower = arena_follower(arena)',
            '        policy_inference.validate_batch(batch_size)\n        net = self.net\n        follower = arena_follower(arena)')
    s = rep(s, '            indptr, indices, values, _own = copies.encode_some(deciding)\n            planes = Planes.from_follower(indptr, indices, values)\n', '')
    s = rep(s, '            step = 4096\n            for start in range(0, count, step):\n                rows = np.arange(start, min(start + step, count))\n                logits = policy_logits(net,\n                    planes.rows(rows).dense(device), torch.from_numpy(allowed[rows]).to(device)\n                )',
            '            for start in range(0, count, batch_size):\n                stop = min(start + batch_size, count)\n                rows = np.arange(start, stop)\n                indptr, indices, values, _own = copies.encode_some(deciding[start:stop])\n                planes = Planes.from_follower(indptr, indices, values)\n                logits = policy_logits(net,\n                    planes.dense(device), torch.from_numpy(allowed[rows]).to(device)\n                )')
    start = s.index('            if len(second):\n                asked = copies.clone_some', s.index('    def play_lookahead'))
    end = s.index('            arena.lookahead_apply(actions.tolist())', start)
    block = s[start:end]
    block = rep(block, '            if len(second):\n',
                '            for start in range(0, len(second), batch_size):\n                group = second[start:start + batch_size]\n')
    block = block.replace('for at in second', 'for at in group').replace('masks[second,', 'masks[group,').replace('len(second)', 'len(group)').replace('actions[second]', 'actions[group]')
    # The outer range still covers the complete set, not a previous group.
    block = block.replace('range(0, len(group), batch_size)', 'range(0, len(second), batch_size)')
    s = s[:start] + block + s[end:]
    return s
edit('neural/contract.py', contract)

def combined(s):
    s = rep(s, 'from __future__ import annotations\n', 'from __future__ import annotations\n\nfrom . import policy_inference\n')
    return rep(s, 'with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):',
               'with policy_inference.autocast(device, self.actions):')
edit('neural/combined.py', combined)

def zoo(s):
    s = rep(s, 'from . import mortal_model\n', 'from . import mortal_model, policy_inference\n')
    s = rep(s, 'with torch.no_grad(), torch.autocast(\n            "cuda", dtype=torch.bfloat16, enabled=str(self.device).startswith("cuda")\n        ):',
            'with torch.no_grad(), policy_inference.autocast(self.device, MORTAL_ACTIONS):', 2)
    s = rep(s, '    logits, _value = player(\n        views.dense(player.kind, rows, players, device),\n        torch.from_numpy(legal).to(device),\n    )',
            '    with policy_inference.autocast(device, int(legal.shape[1])):\n        logits, _value = player(\n            views.dense(player.kind, rows, players, device),\n            torch.from_numpy(legal).to(device),\n        )')
    return s
edit('neural/zoo.py', zoo)

def searched(s):
    s = rep(s, 'from __future__ import annotations\n', 'from __future__ import annotations\n\nfrom . import policy_inference\n')
    s = rep(s, '    objective="hybrid", search_calls=False,\n):',
            '    objective="hybrid", search_calls=False,\n    rollout_batch=policy_inference.DEFAULT_ROLLOUT_BATCH,\n):')
    s = rep(s, '                           evidence=None):',
            '                           evidence=None, rollout_batch=policy_inference.DEFAULT_ROLLOUT_BATCH):')
    s = rep(s, '    extra_candidates: int = 0, audit_share: float = 0.0,\n)',
            '    extra_candidates: int = 0, audit_share: float = 0.0,\n    rollout_batch: int = policy_inference.DEFAULT_ROLLOUT_BATCH,\n)')
    s = rep(s, '    options = dict(margin=margin, hurried=hurried, device=device, pool=pool,',
            '    policy_inference.validate_batch(rollout_batch)\n    options = dict(rollout_batch=rollout_batch, margin=margin, hurried=hurried, device=device, pool=pool,')
    s = rep(s, 'served.play_lookahead(arena, device=device, temperature=temperature)',
            'served.play_lookahead(arena, device=device, temperature=temperature, batch_size=rollout_batch)')
    s = rep(s, 'play_lookahead(net, arena, device=device, temperature=temperature)',
            'play_lookahead(net, arena, device=device, temperature=temperature, step=rollout_batch)')
    s = rep(s, '    validate_budget(games, max_steps)',
            '    policy_inference.validate_batch(rollout_batch)\n    validate_budget(games, max_steps)')
    s = rep(s, 'confirm_worlds=confirm_worlds, evidence=evidence,',
            'confirm_worlds=confirm_worlds, evidence=evidence, rollout_batch=rollout_batch,')
    s = rep(s, '        "leaf_batch": args.leaf_batch,',
            '        "leaf_batch": args.leaf_batch,\n        "rollout_batch": args.rollout_batch,\n        "policy_inference": policy_inference.describe(args.device, served.contract.answers),')
    s = rep(s, '            leaf_batch=args.leaf_batch,',
            '            leaf_batch=args.leaf_batch, rollout_batch=args.rollout_batch,')
    return s
edit('neural/searched.py', searched)

def options(s):
    s = rep(s, '                      extra_candidates=0, audit_share=0.0):',
            '                      extra_candidates=0, audit_share=0.0, rollout_batch=256):')
    s = rep(s, '    if objective not in ("hybrid", "placement"):',
            '    if type(rollout_batch) is not int or rollout_batch <= 0:\n        raise ValueError("rollout_batch must be a positive integer")\n    if objective not in ("hybrid", "placement"):')
    return rep(s, 'def add_arguments(parser):\n',
               'def add_arguments(parser):\n    parser.add_argument("--rollout-batch", type=int, default=256,\n                        help="maximum imagined policy decisions per forward, including riichi")\n')
edit('neural/teacher_options.py', options)

def collect(s):
    s = rep(s, 'from __future__ import annotations\n', 'from __future__ import annotations\n\nfrom . import policy_inference\n')
    s = rep(s, 'VERSION, TEACHER_VERSION, DENSE,', 'VERSION, TEACHER_VERSION, INFERENCE_VERSION, DENSE,')
    s = rep(s, '    sure: float = 1.0\n', '    sure: float = 1.0\n    rollout_batch: int = 256\n')
    s = rep(s, '             head_provenance: dict | None = None) -> dict:',
            '             head_provenance: dict | None = None, device: str = "cpu") -> dict:')
    s = rep(s, '"version": TEACHER_VERSION, "complete": True', '"version": INFERENCE_VERSION, "complete": True')
    s = rep(s, '"placement_head": head_provenance,',
            '"placement_head": head_provenance,\n                    "policy_inference": policy_inference.describe(device, ACTIONS),')
    s = rep(s, '                 head_provenance=provenance)', '                 head_provenance=provenance, device=device)')
    s = rep(s, '                confirm_worlds=settings.confirm_worlds,',
            '                confirm_worlds=settings.confirm_worlds, rollout_batch=settings.rollout_batch,')
    s = rep(s, 'after_logits, _v, _hands = net.everything(after, after_legal)',
            'after_logits, _v, _hands = policy_inference.everything(net, after, after_legal)')
    s = s.replace('"extra_candidates", "audit_share")})', '"extra_candidates", "audit_share", "rollout_batch")})')
    return s
edit('neural/collect_search.py', collect)

def replay(s):
    s = rep(s, 'TEACHER_VERSION = 2\n', 'TEACHER_VERSION = 2\nINFERENCE_VERSION = 3  # Explicit acting precision, tie order and rollout budget.\n')
    s = rep(s, 'not in (VERSION, TEACHER_VERSION)', 'not in (VERSION, TEACHER_VERSION, INFERENCE_VERSION)')
    s = s.replace('== TEACHER_VERSION', 'in (TEACHER_VERSION, INFERENCE_VERSION)')
    insertion = '''        inference = teacher.get("policy_inference")
        if m["version"] == INFERENCE_VERSION:
            if (not isinstance(inference, dict)
                    or set(inference) != {"version", "precision", "tie_break"}
                    or type(inference.get("version")) is not int or inference["version"] != 1
                    or inference.get("precision") not in ("float32", "bfloat16")
                    or inference.get("tie_break") != "first_policy_index"):
                raise ValueError("incomplete acting-policy inference contract")
            _integer(s.get("rollout_batch"), "rollout_batch", 1)
        elif inference is not None or "rollout_batch" in s:
            raise ValueError("new inference contract cannot be downgraded to old teacher replay")
'''
    return rep(s, '        head = teacher.get("placement_head")\n', insertion + '        head = teacher.get("placement_head")\n')
edit('neural/search_replay.py', replay)
edit('neural/train_search.py', lambda s: rep(s, 'if m["version"] == 2:', 'if m["version"] in (2, 3):'))

# Keep remote callers and their immutable experiment identities aligned.
def modal(s):
    begin = s.index('\ndef searched(')
    stop = s.find('\n@app.', begin + 1)
    if stop < 0:
        stop = len(s)
    part = s[begin:stop]
    part = rep(part, '    audit_share: float = 0.0,', '    audit_share: float = 0.0,\n    rollout_batch: int = 256,')
    part = rep(part, 'extra_candidates=extra_candidates, audit_share=audit_share)',
               'extra_candidates=extra_candidates, audit_share=audit_share, rollout_batch=rollout_batch)')
    # Extend the existing forwarding and identity at their explicit audit option.
    part = rep(part, '"--audit-share", str(audit_share)',
               '"--audit-share", str(audit_share), "--rollout-batch", str(rollout_batch)')
    part = rep(part, '"audit_share": audit_share,',
               '"audit_share": audit_share, "rollout_batch": rollout_batch,\n            "policy_inference": {"version": 1, "precision": "bfloat16", "tie_break": "first_policy_index"},')
    return s[:begin] + part + s[stop:]
edit('neural/modal_app.py', modal)

# The runtime memory regression allocates a new legality tensor on every load.
def memory_test(s):
    s = rep(s, '    let session, input;\n', '    let session, input, mask;\n')
    s = rep(s, '      const mask = wants\n', '      mask = wants\n')
    return rep(s, '      input?.dispose();\n', '      input?.dispose();\n      mask?.dispose();\n')
edit('web/tests/runtime-memory.test.js', memory_test)

# Exercise exported one/two-input models with the real reduced WASM runtime.
def ci(s):
    marker = '      - "web/runtime/reduced-ops.config"\n'
    s = rep(s, marker, marker + '      - "web/src/lib/policy.worker.js"\n      - "web/tests/policy-export-parity.mjs"\n      - "web/public/ort/**"\n', 2)
    return s + '''      - uses: actions/setup-node@820762786026740c76f36085b0efc47a31fe5020
        with:
          node-version: "22"
      - name: Test exported policies through the worker and reduced WASM runtime
        run: |
          npm ci --prefix web
          python -m neural.tests.export_worker_fixture "$RUNNER_TEMP/policy-fixture"
          node web/tests/policy-export-parity.mjs "$RUNNER_TEMP/policy-fixture"
'''
edit('.github/workflows/neural.yml', ci)
Path('.review/changed.txt').write_text('\n'.join(changed) + '\n')
print('Guarded edits:', '\n'.join(changed))
