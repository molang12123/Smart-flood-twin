"""Fresh-process integration: actual CLI, checkpoint, gate, and all export artifacts."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class WorkflowTests(unittest.TestCase):
    def test_train_then_independent_predict_and_manifest(self):
        cfg = json.loads((ROOT/'configs/synthetic_parametric.json').read_text())
        for key in ('csv','metadata','roughness'):
            cfg['terrain'][key] = str((ROOT/'configs'/cfg['terrain'][key]).resolve())
        cfg['reference_csv'] = str((ROOT/'configs'/cfg['reference_csv']).resolve())
        cfg['training'].update(adam_steps=2,lbfgs_steps=0,max_seconds=30)
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d); config=folder/'config.json'
            config.write_text(json.dumps(cfg))
            def cli(*args):
                return subprocess.run([sys.executable,'-m','flood_pinn',*map(str,args)],
                                      cwd=ROOT,text=True,capture_output=True,timeout=60)
            trained=cli('train',config,'--output',folder/'training')
            self.assertEqual(trained.returncode,0,trained.stderr)
            history=folder/'training/training_history.csv'
            original=history.read_bytes()
            base=['predict',config,'--model',folder/'training/model','--rainfall-mm-hr','60',
                  '--duration-min',str(3.5/60),'--times','0','3.5','6','--output',folder/'prediction']
            rejected=cli(*base)
            self.assertEqual(rejected.returncode,2)
            self.assertEqual(json.loads(rejected.stderr)['code'],'UNVALIDATED_SCENARIO')
            inferred=cli(*base,'--allow-unvalidated')
            self.assertEqual(inferred.returncode,0,inferred.stderr)
            self.assertEqual(history.read_bytes(),original)
            summary=json.loads((folder/'prediction/simulation_summary.json').read_text())
            self.assertEqual(summary['validation_status'],'research_unvalidated')
            self.assertEqual(summary['total_simulation_time_s'],6)
            manifest=json.loads((folder/'prediction/manifest.json').read_text())
            for entry in manifest['files']:
                path=folder/'prediction'/entry['path']
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),entry['sha256'])
                if entry['row_count'] is not None:
                    with path.open(newline='') as stream:
                        self.assertEqual(len(list(csv.DictReader(stream))),entry['row_count'])
            self.assertTrue((folder/'prediction/benchmark_report.json').is_file())

    def test_malformed_config_error(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.json';p.write_text('[]')
            result=subprocess.run([sys.executable,'-m','flood_pinn','validate-input',str(p)],
                                  cwd=ROOT,text=True,capture_output=True,timeout=30)
            self.assertEqual(result.returncode,2)
            self.assertEqual(json.loads(result.stderr)['code'],'WORKFLOW_ERROR')
