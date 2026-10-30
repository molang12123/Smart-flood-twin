"""Editor workflow: fixed-file identity, manual rain, safe errors and report output."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from flood_pinn.desktop import load_settings, run_workflow

ROOT = Path(__file__).resolve().parents[1]


class DesktopTests(unittest.TestCase):
    def _fixture(self, folder):
        base = json.loads((ROOT/'configs/fixed_terrain.json').read_text())
        for key in ('csv','metadata','roughness'):
            base['terrain'][key] = str((ROOT/'configs'/base['terrain'][key]).resolve())
        base['training'].update(adam_steps=1,lbfgs_steps=0,max_seconds=30)
        base['model'].update(width=8,layers=2)
        (folder/'base.json').write_text(json.dumps(base))
        settings = dict(action='train_and_predict',base_config='base.json',
                        rainfall_mm_hr=60,duration=dict(value=.05,unit='min'),
                        output_root='results',model_dir=None,allow_unvalidated=True,
                        batch_size=32,open_report=False)
        path=folder/'run.json';path.write_text(json.dumps(settings))
        return base, settings, path

    def test_train_and_predict_preserves_terrain_and_reports_manual_rain(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);base,settings,path=self._fixture(folder)
            source=Path(base['terrain']['csv'])
            before=hashlib.sha256(source.read_bytes()).hexdigest()
            result=run_workflow(path)
            self.assertEqual(result['status'],'complete',result.get('error'))
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),before)
            self.assertTrue(result['report_path'].is_file())
            summary=json.loads((result['run_dir']/'prediction/simulation_summary.json').read_text())
            self.assertEqual(summary['rainfall_mm_hr'],60)
            self.assertEqual(summary['rain_duration_s'],3.)
            metadata=json.loads((result['run_dir']/'prediction/grid_metadata.json').read_text())
            self.assertEqual(metadata['fingerprint'],result['terrain_fingerprint'])
            # Change only manually supplied rain, load existing model: must not train.
            settings.update(action='predict',model_dir=str(result['run_dir']/'training/model'),rainfall_mm_hr=40)
            path.write_text(json.dumps(settings))
            with patch('flood_pinn.training.train',side_effect=AssertionError('prediction retrained')):
                second=run_workflow(path)
            self.assertEqual(second['status'],'complete',second.get('error'))
            self.assertNotEqual(second['run_dir'],result['run_dir'])
            self.assertEqual(second['terrain_fingerprint'],result['terrain_fingerprint'])
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),before)
            self.assertFalse((second['run_dir']/'training/training_history.csv').exists())

    def test_out_of_range_rejected_before_training_and_failure_report_written(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);_,settings,path=self._fixture(folder)
            settings['rainfall_mm_hr']=900;path.write_text(json.dumps(settings))
            with patch('flood_pinn.training.train',side_effect=AssertionError('should not train')):
                result=run_workflow(path)
            self.assertEqual(result['status'],'failed')
            self.assertIn('训练范围',result['error'])
            self.assertTrue(result['report_path'].is_file())
            self.assertFalse((result['run_dir']/'training').exists())

    def test_settings_validation_does_not_guess_units_or_booleans(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);_,settings,path=self._fixture(folder)
            for key,value in [('duration',{'value':3,'unit':'minutes'}),('allow_unvalidated','false'),
                              ('rainfall_mm_hr',float('nan')),('batch_size',False)]:
                bad=dict(settings);bad[key]=value;path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):load_settings(path)

    def test_bad_settings_or_output_path_still_produce_failure_report(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);_,settings,path=self._fixture(folder)
            path.write_text('{broken json')
            with patch('flood_pinn.desktop.ROOT',folder):
                bad_json=run_workflow(path)
            self.assertEqual(bad_json['status'],'failed')
            self.assertTrue(bad_json['report_path'].is_file())
            (folder/'not_a_directory').write_text('preserve')
            settings['output_root']='not_a_directory';path.write_text(json.dumps(settings))
            with patch('flood_pinn.desktop.ROOT',folder):
                bad_path=run_workflow(path)
            self.assertEqual(bad_path['status'],'failed')
            self.assertTrue(bad_path['report_path'].is_file())
            self.assertEqual((folder/'not_a_directory').read_text(),'preserve')

    def test_vscode_launcher_uses_debug_console_and_real_entry(self):
        launcher=json.loads((ROOT/'.vscode/launch.json').read_text())['configurations'][0]
        self.assertEqual(launcher['console'],'internalConsole')
        self.assertTrue((ROOT/'run_in_vscode.py').is_file())
        self.assertIn('vscode_run.json',' '.join(launcher['args']))


if __name__ == '__main__':unittest.main()
