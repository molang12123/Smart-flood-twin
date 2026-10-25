import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from flood_pinn.terrain import TerrainGrid


class TerrainTests(unittest.TestCase):
    def fixture(self,path):
        grid=TerrainGrid.synthetic(nx=4,ny=3,kind='slope')
        csv_path=path/'terrain.csv'; metadata_path=path/'terrain_metadata.json'
        with csv_path.open('w',newline='') as f:
            w=csv.writer(f); w.writerow(['cell_id','row','col','x','y','z'])
            for cell,(x,y,z) in enumerate(zip(grid.x,grid.y,grid.z.ravel())):
                w.writerow([cell,cell//grid.nx,cell%grid.nx,x,y,z])
        metadata=grid.metadata.copy()
        metadata['sha256']=hashlib.sha256(csv_path.read_bytes()).hexdigest()
        metadata_path.write_text(json.dumps(metadata))
        return grid,csv_path,metadata_path

    def test_slope_derivative_and_outer_edges(self):
        grid=TerrainGrid.synthetic(nx=4,ny=3,kind='slope')
        x=np.array([0,10,35,80]); y=np.array([0,20,35,60])
        q=grid.query(x,y)
        np.testing.assert_allclose(q['z'],10-.01*x+.005*y,atol=1e-14)
        np.testing.assert_allclose(q['zx'],-.01,atol=1e-14)
        np.testing.assert_allclose(q['zy'],.005,atol=1e-14)
        with self.assertRaises(ValueError): grid.query(81,0)

    def test_file_roundtrip_hash_and_roughness(self):
        with tempfile.TemporaryDirectory() as d:
            original,csv_path,metadata_path=self.fixture(Path(d))
            grid=TerrainGrid.from_files(csv_path,metadata_path,manning_n=.02)
            np.testing.assert_array_equal(grid.points,original.points)
            np.testing.assert_array_equal(grid.z,original.z)
            with self.assertRaises(ValueError): TerrainGrid.from_files(csv_path,metadata_path)
            with csv_path.open('a') as f: f.write('\n')
            with self.assertRaisesRegex(ValueError,'SHA'): TerrainGrid.from_files(csv_path,metadata_path,manning_n=.02)

    def test_duplicate_missing_and_wrong_centres(self):
        for mode in ['duplicate','missing','coordinate','nan']:
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as d:
                _,csv_path,metadata_path=self.fixture(Path(d))
                metadata=json.loads(metadata_path.read_text()); metadata.pop('sha256');metadata_path.write_text(json.dumps(metadata))
                with csv_path.open() as f: rows=list(csv.reader(f))
                if mode=='duplicate': rows[2]=rows[1]
                elif mode=='missing': rows.pop()
                elif mode=='coordinate': rows[1][3]='12345'
                else: rows[1][5]='nan'
                with csv_path.open('w',newline='') as f: csv.writer(f).writerows(rows)
                with self.assertRaises(ValueError): TerrainGrid.from_files(csv_path,metadata_path,manning_n=.02)

    def test_metadata_and_nodata_failures(self):
        grid=TerrainGrid.synthetic(nx=3,ny=3)
        for key,value in [('horizontal_unit','degrees'),('vertical_datum',None),('extent',[0,1,0,1]),('synthetic',False),('domain_type','masked')]:
            meta=grid.metadata.copy();meta[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError): TerrainGrid(meta,grid.z,grid.n)
        meta=grid.metadata.copy();meta['nodata_value']=grid.z[0,0]
        with self.assertRaises(ValueError): TerrainGrid(meta,grid.z,grid.n)

    def test_real_crs_checks(self):
        grid=TerrainGrid.synthetic(nx=3,ny=3)
        meta=grid.metadata.copy();meta.update(synthetic=False,crs='EPSG:4326',crs_type='projected')
        with self.assertRaises(ValueError): TerrainGrid(meta,grid.z,grid.n)
        meta['crs']='EPSG:3826'
        self.assertEqual(TerrainGrid(meta,grid.z,grid.n).nx,3)
        meta['crs']='nonsense'
        with self.assertRaises(ValueError): TerrainGrid(meta,grid.z,grid.n)

    def test_roughness_index_and_hash_validation(self):
        with tempfile.TemporaryDirectory() as d:
            _,csv_path,metadata_path=self.fixture(Path(d))
            roughness=Path(d)/'roughness.csv'
            roughness.write_text('cell_id,manning_n\n'+''.join(f'{i},0.02\n' for i in range(12)))
            TerrainGrid.from_files(csv_path,metadata_path,roughness_path=roughness)
            roughness.write_text(roughness.read_text().replace('1,0.02','0,0.02',1))
            with self.assertRaises(ValueError): TerrainGrid.from_files(csv_path,metadata_path,roughness_path=roughness)
            metadata=json.loads(metadata_path.read_text());metadata['roughness_sha256']='0'*64
            metadata_path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError,'SHA'): TerrainGrid.from_files(csv_path,metadata_path,roughness_path=roughness)

    def test_bilinear_derivatives_match_surface(self):
        grid=TerrainGrid.synthetic(nx=6,ny=5)
        x=np.array([13.,37.,71.]); y=np.array([17.,45.,79.]); step=1e-4
        q=grid.query(x,y)
        np.testing.assert_allclose(q['zx'],(grid.query(x+step,y)['z']-grid.query(x-step,y)['z'])/(2*step),atol=1e-10)
        np.testing.assert_allclose(q['zy'],(grid.query(x,y+step)['z']-grid.query(x,y-step)['z'])/(2*step),atol=1e-10)

    def test_undulating_slope_convergence(self):
        errors=[]
        for nx in [10,20,40]:
            grid=TerrainGrid.synthetic(nx=nx,ny=nx,dx=100/nx,dy=100/nx)
            x,y=37.3,42.7
            expected=-.005+.3*(2*np.pi/100)*np.cos(2*np.pi*x/100)*np.cos(2*np.pi*y/100)
            errors.append(abs(grid.query(x,y)['zx']-expected))
        self.assertLess(errors[-1],errors[0])

    def test_immutable_checkpoint_identity(self):
        grid=TerrainGrid.synthetic(nx=2,ny=2)
        fingerprint=grid.fingerprint
        for name in ['z','n','x','y']:
            with self.subTest(name=name):
                array=getattr(grid,name)
                with self.assertRaises(ValueError): array.flat[0]=999
                with self.assertRaises(ValueError): array.setflags(write=True)
                with self.assertRaises(AttributeError): setattr(grid,name,array.copy())
        with self.assertRaises(AttributeError): grid.dx=30
        metadata=grid.metadata;metadata['dx']=30
        metadata['assumptions'].append('changed')
        self.assertEqual(grid.metadata['dx'],20)
        self.assertNotIn('changed',grid.metadata['assumptions'])
        self.assertEqual(grid.fingerprint,fingerprint)

    def test_roughness_edge_extension_positive(self):
        grid=TerrainGrid.synthetic(nx=2,ny=2)
        grid=TerrainGrid(grid.metadata,grid.z,np.array([[.001,.2],[.002,.3]]))
        self.assertGreater(grid.query(0,0)['n'],0)


if __name__=='__main__': unittest.main()
