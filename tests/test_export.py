import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import numpy as np
from flood_pinn.export import export_simulation,refresh_manifest


class Terrain:
    nx=2;ny=2;dx=2.;dy=3.;x_origin=0.;y_origin=0.
    bounds=(0.,4.,0.,6.)
    fingerprint='synthetic-test'
    metadata={'grid_id':'test-grid','synthetic':True}
    z=np.zeros((2,2))
    x=np.array([1.,3.,1.,3.]);y=np.array([1.5,1.5,4.5,4.5])
    def to_metadata(self):return dict(self.metadata,nx=self.nx,ny=self.ny,dx=self.dx,dy=self.dy)


def rain(rate=3600.,duration=.4):
    return SimpleNamespace(intensity_mm_hr=rate,duration_s=duration,
                           integral=lambda a,b: rate/3600000*(min(b,duration)-min(a,duration)))


class ExportTests(unittest.TestCase):
    def test_uniform_rain_ledger_and_roundtrip(self):
        rainfall=rain()
        def evaluate(p): return np.column_stack([.002+.001*np.minimum(p[:,2],.4),np.zeros(len(p)),np.zeros(len(p))])
        with TemporaryDirectory() as temp:
            summary=export_simulation(Terrain(),evaluate,[0,.3,.4,.7],rainfall,temp,make_plots=False)
            self.assertAlmostEqual(summary['water_balance']['error_m3'],0,places=14)
            self.assertAlmostEqual(summary['water_balance']['rain_volume_m3'],.0004*24)
            self.assertEqual(summary['reference_validation_status'],'not_completed')
            records=list(csv.DictReader((Path(temp)/'simulation_results.csv').read_text().splitlines()))
            self.assertEqual(len(records),16)
            self.assertEqual([int(r['cell_id']) for r in records],list(range(4))*4)
            for row in records:
                self.assertAlmostEqual(float(row['water_surface']),float(row['z'])+float(row['h']))
                self.assertEqual(row['direction'],'')
            cells=list(csv.DictReader((Path(temp)/'cell_water_balance.csv').read_text().splitlines()))
            self.assertEqual(len(cells),12)
            self.assertLess(max(abs(float(r['balance_error_m3'])) for r in cells),1e-15)
            loaded=json.loads((Path(temp)/'simulation_summary.json').read_text())
            self.assertEqual(loaded['max_depth_m'],summary['max_depth_m'])
            self.assertAlmostEqual(summary['max_depth_m'],max(float(r['h']) for r in records))
            self.assertEqual(summary['max_depth_time_s'],.4)
    def test_constant_raw_flux_boundary_signs(self):
        evaluate=lambda p: np.tile([1.,2.,0.],(len(p),1))
        with TemporaryDirectory() as temp:
            summary=export_simulation(Terrain(),evaluate,[0,1],rain(0),temp,make_plots=False)
            self.assertAlmostEqual(summary['water_balance']['boundary_in_m3'],12.)
            self.assertAlmostEqual(summary['water_balance']['boundary_out_m3'],12.)
            self.assertAlmostEqual(summary['max_cell_balance_error_m3'],0.)
            records=list(csv.DictReader((Path(temp)/'simulation_results.csv').read_text().splitlines()))
            self.assertTrue(all(float(r['direction'])==0 for r in records))
    def test_dry_display_preserves_raw_and_storage(self):
        evaluate=lambda p: np.tile([.00001,3.,4.],(len(p),1))
        with TemporaryDirectory() as temp:
            summary=export_simulation(Terrain(),evaluate,[0,1],rain(0),temp,make_plots=False)
            records=list(csv.DictReader((Path(temp)/'simulation_results.csv').read_text().splitlines()))
            self.assertTrue(all(r['direction']=='' and float(r['speed'])==0 for r in records))
            self.assertTrue(all(float(r['u_raw'])==3 for r in records))
            self.assertAlmostEqual(summary['water_balance']['initial_storage_m3'],24e-5)
            self.assertAlmostEqual(summary['water_balance']['boundary_in_m3'],.00001*(3*6+4*4))
            self.assertEqual(summary['max_raw_speed_m_s'],5.)
    def test_local_global_ledger_agree_even_when_prediction_is_not_conservative(self):
        def evaluate(p):
            return np.column_stack([1+.1*p[:,2]*(p[:,0]+p[:,1]),p[:,0],np.zeros(len(p))])
        with TemporaryDirectory() as temp:
            summary=export_simulation(Terrain(),evaluate,[0,1],rain(0),temp,make_plots=False)
            cells=list(csv.DictReader((Path(temp)/'cell_water_balance.csv').read_text().splitlines()))
            self.assertAlmostEqual(sum(float(r['balance_error_m3']) for r in cells),summary['water_balance']['error_m3'])
            self.assertGreater(summary['max_global_balance_error_m3'],1.)
            manifest=json.loads((Path(temp)/'manifest.json').read_text())
            state=next(item for item in manifest['files'] if item['path']=='simulation_results.csv')
            self.assertEqual(state['row_count'],8)
            self.assertEqual(len(state['sha256']),64)
    def test_invalid_fields_fail_without_partial_normal_outputs(self):
        for field in [[float('nan'),0,0],[-1,0,0],[1,float('inf'),0]]:
            with self.subTest(field=field),TemporaryDirectory() as temp:
                with self.assertRaises(FloatingPointError):
                    export_simulation(Terrain(),lambda p:np.tile(field,(len(p),1)),[0,1],rain(0),temp,make_plots=False)
                self.assertFalse((Path(temp)/'simulation_results.csv').exists())
                self.assertFalse((Path(temp)/'.export.lock').exists())
                self.assertEqual(json.loads((Path(temp)/'simulation_failure.json').read_text())['status'],'failed')
    def test_manifest_refresh_and_declared_end_rain(self):
        import hashlib
        evaluate=lambda p:np.tile([1.,1e-11,0.],(len(p),1))
        with TemporaryDirectory() as temp:
            summary=export_simulation(Terrain(),evaluate,[0,.1],rain(),temp,
                                      run_metadata={'end_s':.7,'coverage':'partial'},make_plots=False)
            self.assertAlmostEqual(summary['total_simulation_time_s'],.7)
            self.assertAlmostEqual(summary['total_rain_volume_m3'],.0004*24)
            self.assertAlmostEqual(summary['rain_volume_through_last_frame_m3'],.0001*24)
            self.assertAlmostEqual(summary['accounted_rain_volume_m3'],.0001*24)
            records=list(csv.DictReader((Path(temp)/'simulation_results.csv').read_text().splitlines()))
            self.assertTrue(all(r['direction']=='' for r in records))
            summary['new_timing_s']=.1
            path=Path(temp)/'simulation_summary.json';path.write_text(json.dumps(summary))
            (Path(temp)/'benchmark_report.json').write_text('{}')
            manifest=refresh_manifest(temp)
            item=next(item for item in manifest['files'] if item['path']==path.name)
            self.assertEqual(item['sha256'],hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(next(item for item in manifest['files'] if item['path']=='simulation_results.csv')['row_count'],8)
            self.assertIn('benchmark_report.json',[item['path'] for item in manifest['files']])
    def test_nonresult_files_also_protected(self):
        evaluate=lambda p:np.tile([1.,0.,0.],(len(p),1))
        with TemporaryDirectory() as temp:
            path=Path(temp)/'grid_metadata.json';path.write_text('user content')
            with self.assertRaises(FileExistsError):
                export_simulation(Terrain(),evaluate,[0],rain(0),temp,make_plots=False)
            self.assertEqual(path.read_text(),'user content')
    def test_preexisting_lock_is_preserved_and_refuses_export(self):
        evaluate=lambda p:np.tile([1.,0.,0.],(len(p),1))
        with TemporaryDirectory() as temp:
            lock=Path(temp)/'.export.lock';lock.write_text('other owner')
            with self.assertRaises(FileExistsError):
                export_simulation(Terrain(),evaluate,[0,1],rain(0),temp,make_plots=False)
            self.assertEqual(lock.read_text(),'other owner')
            self.assertFalse((Path(temp)/'simulation_results.csv').exists())
    def test_reentrant_export_cannot_clobber_outer_run(self):
        with TemporaryDirectory() as temp:
            attempted=False
            def evaluate(p):
                nonlocal attempted
                self.assertTrue((Path(temp)/'.export.lock').exists())
                if not attempted:
                    attempted=True
                    with self.assertRaises(FileExistsError):
                        export_simulation(Terrain(),evaluate,[0,1],rain(0),temp,make_plots=False)
                return np.tile([1.,0.,0.],(len(p),1))
            export_simulation(Terrain(),evaluate,[0,1],rain(0),temp,make_plots=False)
            self.assertTrue(attempted)
            self.assertFalse((Path(temp)/'.export.lock').exists())
            manifest=json.loads((Path(temp)/'manifest.json').read_text())
            self.assertNotIn('.export.lock',[item['path'] for item in manifest['files']])
            lock=Path(temp)/'.export.lock';lock.write_text('test refresh exclusion')
            refreshed=refresh_manifest(temp)
            self.assertNotIn('.export.lock',[item['path'] for item in refreshed['files']])
            lock.unlink()
    def test_late_external_file_is_not_overwritten(self):
        with TemporaryDirectory() as temp:
            changed=False
            def evaluate(p):
                nonlocal changed
                if not changed:
                    changed=True
                    (Path(temp)/'grid_metadata.json').write_text('external content')
                return np.tile([1.,0.,0.],(len(p),1))
            with self.assertRaises(FileExistsError):
                export_simulation(Terrain(),evaluate,[0,1],rain(0),temp,make_plots=False)
            self.assertEqual((Path(temp)/'grid_metadata.json').read_text(),'external content')
            self.assertFalse((Path(temp)/'.export.lock').exists())
            self.assertFalse((Path(temp)/'simulation_results.csv').exists())
    def test_time_validation_and_existing_run_protection(self):
        evaluate=lambda p:np.tile([1.,0.,0.],(len(p),1))
        with TemporaryDirectory() as temp:
            for times in [[],[1,0],[0,0],[float('inf')]]:
                with self.assertRaises(ValueError):export_simulation(Terrain(),evaluate,times,rain(0),temp,make_plots=False)
            export_simulation(Terrain(),evaluate,[0],rain(0),temp,make_plots=False)
            with self.assertRaises(FileExistsError):export_simulation(Terrain(),evaluate,[0],rain(0),temp,make_plots=False)
