"""Publication is bound to evaluated bytes, not a mutable candidate pathname."""
from contextlib import redirect_stdout
import hashlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from neural import promote, gate
from neural.checkpoints import atomic_save


class PromotionRepairs(unittest.TestCase):
    def test_changed_candidate_during_evaluation_does_not_change_promoted_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);candidate=root/'latest.pt';champion=root/'previous.pt';out=root/'out'
            atomic_save({'generation':1},candidate);atomic_save({'generation':0},champion)
            expected=candidate.read_bytes()
            def compare(snapshot,old,**kwargs):
                self.assertEqual(snapshot.read_bytes(),expected)
                atomic_save({'generation':2},candidate)
                return {'promote':True,'candidate':{'sha256':hashlib.sha256(snapshot.read_bytes()).hexdigest()},
                        'champion':{'sha256':hashlib.sha256(old.read_bytes()).hexdigest()}}
            with patch.object(sys,'argv',['promote',str(candidate),str(champion),'--out',str(out),'--seed','900001']), \
                 patch.object(gate,'compare',side_effect=compare),redirect_stdout(io.StringIO()):
                promote.main()
            self.assertEqual((out/'champion.pt').read_bytes(),expected)
            self.assertEqual(torch.load(candidate,weights_only=True)['generation'],2)
            self.assertEqual(len(list((out/'promotion-reports').glob('*.json'))),1)

    def test_hash_mismatch_and_rejection_preserve_champion(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);snapshot=root/'candidate.pt';destination=root/'champion.pt'
            atomic_save({'generation':1},snapshot);atomic_save({'generation':0},destination)
            before=destination.read_bytes()
            for report in ({'promote':False},{'promote':True,'candidate':{'sha256':'0'*64}}):
                with self.assertRaises(ValueError):promote.publish_evaluated_snapshot(snapshot,root,report)
                self.assertEqual(destination.read_bytes(),before)

    def test_legacy_writer_is_atomic_on_partial_serialization_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);atomic_save({'generation':0},root/'champion.pt');before=(root/'champion.pt').read_bytes()
            verdict=promote.Verdict(True,'test',2.,'candidate','champion',1)
            def broken(payload,stream):
                stream.write(b'partial serialization');raise OSError('injected save failure')
            with patch.object(torch,'save',side_effect=broken),self.assertRaises(OSError):
                promote.promote({'generation':1},root,verdict)
            self.assertEqual((root/'champion.pt').read_bytes(),before)


if __name__=='__main__':unittest.main()
