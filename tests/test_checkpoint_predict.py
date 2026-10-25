import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from flood_pinn.checkpoint import save_checkpoint, load_checkpoint
from flood_pinn.contracts import PredictionError
from flood_pinn.model import ParametricPINN
from flood_pinn.predict import Predictor
from flood_pinn.terrain import TerrainGrid


def fixture(grid):
    scales = dict(x_origin=grid.x_origin,y_origin=grid.y_origin,x_scale=grid.nx*grid.dx,
                  y_scale=grid.ny*grid.dy,t_scale=6.,r_scale=1e-4,duration_scale=6.,h_scale=.001,u_scale=.01)
    config = dict(scales=scales,model=dict(width=8,layers=2,parameterized=True),
                  conditions=dict(initial="dry",boundaries=dict.fromkeys(["west","east","south","north"],"wall")),
                  time=dict(end_s=6.,output_interval_s=2.),scenarios=[
                      dict(scenario_id="a",rainfall_mm_hr=20.,duration_s=2.,split="train"),
                      dict(scenario_id="b",rainfall_mm_hr=80.,duration_s=4.,split="train")])
    return ParametricPINN(scales,width=8,layers=2),config


class CheckpointTests(unittest.TestCase):
    def test_malformed_metadata_is_structured(self):
        grid=TerrainGrid.synthetic()
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/'model_metadata.json').write_text('[]')
            with self.assertRaises(PredictionError) as caught: load_checkpoint(d,grid)
            self.assertEqual(caught.exception.code,'MODEL_INCOMPATIBLE')

    def test_roundtrip_and_wrong_identity(self):
        grid=TerrainGrid.synthetic(kind="flat")
        model, config=fixture(grid)
        with tempfile.TemporaryDirectory() as d:
            save_checkpoint(model,grid,config,d,{"training_completed": True})
            loaded,metadata=load_checkpoint(d,grid)
            p=torch.tensor([[20.,30.,1.,1e-5,2.]])
            torch.testing.assert_close(model(p),loaded(p),rtol=0,atol=0)
            with self.assertRaises(PredictionError) as caught:
                load_checkpoint(d,TerrainGrid.synthetic(kind="slope"))
            self.assertEqual(caught.exception.code,"TERRAIN_VERSION_MISMATCH")
            changed=copy.deepcopy(config["conditions"]); changed["initial"]="wet"
            with self.assertRaises(PredictionError): load_checkpoint(d,grid,conditions=changed)
            (Path(d)/"weights.pt").write_bytes(b"corrupt")
            with self.assertRaises(PredictionError): load_checkpoint(d,grid)

    def test_research_gate_streaming_and_json(self):
        grid=TerrainGrid.synthetic(nx=2,ny=2,kind="flat")
        model,config=fixture(grid)
        with tempfile.TemporaryDirectory() as d:
            save_checkpoint(model,grid,config,d,{"training_completed":True})
            predictor=Predictor.from_checkpoint(d,grid)
            with self.assertRaises(PredictionError) as caught:
                predictor.predict(50.,.05,[0,3,6])
            self.assertEqual(caught.exception.code,"UNVALIDATED_SCENARIO")
            result=predictor.predict(50.,.05,[3.,1.,3.],output_mode="requested",allow_unvalidated=True)
            self.assertEqual(result.finalize()["status"],"pending")
            frames=list(result.frames_iterator)
            self.assertEqual([f["t"] for f in frames],[1.,3.])
            json.dumps(frames,allow_nan=False)
            self.assertEqual(result.finalize()["status"],"complete")
            self.assertEqual(result.validation_status,"research_unvalidated")
            with self.assertRaises(PredictionError):
                predictor.predict(900.,.05,[1],allow_unvalidated=True)
            with self.assertRaises(PredictionError):
                predictor.predict(50.,.05,[9],allow_unvalidated=True)

    def test_cannot_promote_smoke_model_to_validated(self):
        grid=TerrainGrid.synthetic(); model,config=fixture(grid)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                save_checkpoint(model,grid,config,d,{"training_completed":True},
                    {"mode":"discrete","scenarios":[dict(rainfall_mm_hr=50.,duration_s=3.)],"evidence_id":"missing"})

    def test_legacy_single_scenario_cannot_change_rain(self):
        grid=TerrainGrid.synthetic(); _,config=fixture(grid)
        config['scenarios']=config['scenarios'][:1]
        config['model']['parameterized']=False
        model=ParametricPINN(config['scales'],width=8,layers=2,parameterized=False)
        with tempfile.TemporaryDirectory() as d:
            save_checkpoint(model,grid,config,d,{})
            p=Predictor.from_checkpoint(d,grid)
            with self.assertRaises(PredictionError): p.predict(80.,4/60,[1],allow_unvalidated=True)


if __name__ == "__main__": unittest.main()
