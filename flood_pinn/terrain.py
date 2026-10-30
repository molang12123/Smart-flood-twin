"""Strict, QGIS-independent reader for already standardized metre grids."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


class TerrainGrid:
    """Complete rectangular grid; no GIS repair, reprojection or NoData filling."""

    def __init__(self, metadata, z, n):
        if not isinstance(metadata, dict):
            raise ValueError('INPUT_DATA_INVALID: metadata must be a JSON object')
        for name in ['schema_version','grid_id','terrain_id','terrain_version']:
            if not isinstance(metadata.get(name),str) or not metadata[name].strip():
                raise ValueError(f'INPUT_DATA_INVALID: {name} is required')
        if not isinstance(metadata.get('synthetic'),bool):
            raise ValueError('INPUT_DATA_INVALID: synthetic must be explicitly true or false')
        self._metadata = copy.deepcopy(metadata)
        self.nx = _integer(metadata.get('nx'), 'nx')
        self.ny = _integer(metadata.get('ny'), 'ny')
        if self.nx < 2 or self.ny < 2:
            raise ValueError('UNSUPPORTED_DOMAIN: nx and ny must each be at least two')
        self.dx = _positive(metadata.get('dx'), 'dx')
        self.dy = _positive(metadata.get('dy'), 'dy')
        self.x_origin = _finite(metadata.get('x_edge_origin'), 'x_edge_origin')
        self.y_origin = _finite(metadata.get('y_edge_origin'), 'y_edge_origin')
        if metadata.get('horizontal_unit') != 'm' or metadata.get('vertical_unit') != 'm':
            raise ValueError('INPUT_DATA_INVALID: horizontal_unit and vertical_unit must be m')
        if not metadata.get('vertical_datum'):
            raise ValueError('INPUT_DATA_INVALID: vertical_datum is required')
        crs = metadata.get('crs')
        if crs == 'LOCAL_ENGINEERING':
            if metadata.get('synthetic') is not True:
                raise ValueError('INPUT_DATA_INVALID: LOCAL_ENGINEERING requires synthetic=true')
        else:
            _validate_projected_crs(crs, metadata.get('crs_type'))
        if metadata.get('axis_order', 'x_east_y_north') != 'x_east_y_north':
            raise ValueError('INPUT_DATA_INVALID: axis_order must be x_east_y_north')
        if metadata.get('row_order', 'south_to_north') != 'south_to_north':
            raise ValueError('INPUT_DATA_INVALID: row_order must be south_to_north')
        if not metadata.get('nodata_policy'):
            raise ValueError('INPUT_DATA_INVALID: nodata_policy is required')
        if metadata.get('domain_type', 'rectangle') != 'rectangle':
            raise ValueError('UNSUPPORTED_DOMAIN: only complete rectangular domains are supported')
        self.bounds = (self.x_origin, self.x_origin + self.nx*self.dx,
                       self.y_origin, self.y_origin + self.ny*self.dy)
        extent = metadata.get('extent')
        if not isinstance(extent, (list, tuple)) or len(extent) != 4 or not np.allclose(
                np.asarray(extent, dtype=float), self.bounds, rtol=0, atol=1e-7):
            raise ValueError('INPUT_DATA_INVALID: extent must be [xmin,xmax,ymin,ymax] outer edges')
        self._z = np.asarray(z, dtype=np.float64).copy()
        self._n = np.asarray(n, dtype=np.float64).copy()
        if self.z.shape != (self.ny, self.nx) or self.n.shape != self.z.shape:
            raise ValueError('INPUT_DATA_INVALID: terrain/roughness shape mismatch')
        if not np.isfinite(self.z).all() or not np.isfinite(self.n).all() or (self.n <= 0).any():
            raise ValueError('INPUT_DATA_INVALID: nonfinite elevation or nonpositive/nonfinite roughness')
        nodata = metadata.get('nodata_value')
        if nodata is not None and np.any(self._z == float(nodata)):
            raise ValueError('INPUT_DATA_INVALID: NoData inside complete domain')
        xx, yy = np.meshgrid(self.x_origin+(np.arange(self.nx)+.5)*self.dx,
                             self.y_origin+(np.arange(self.ny)+.5)*self.dy)
        self._x, self._y = xx.ravel().copy(), yy.ravel().copy()
        digest = hashlib.sha256()
        digest.update(json.dumps(self.metadata, sort_keys=True, allow_nan=False).encode())
        digest.update(self.z.astype('<f8').tobytes())
        digest.update(self.n.astype('<f8').tobytes())
        self.fingerprint = digest.hexdigest()
        for array in [self._z,self._n,self._x,self._y]:
            array.setflags(write=False)
        self._sealed = True

    def __setattr__(self,name,value):
        if getattr(self,'_sealed',False) and not name.startswith('_'):
            raise AttributeError('TerrainGrid is immutable; construct a new grid for changed data')
        object.__setattr__(self,name,value)

    @property
    def metadata(self):
        # Callers may annotate their copy, never alter the checkpoint identity in place.
        return copy.deepcopy(self._metadata)

    @property
    def z(self):
        return self._z.view()

    @property
    def n(self):
        return self._n.view()

    @property
    def x(self):
        return self._x.view()

    @property
    def y(self):
        return self._y.view()

    @property
    def points(self):
        return np.column_stack((self.x, self.y))

    def to_metadata(self):
        result = copy.deepcopy(self.metadata)
        result['fingerprint'] = self.fingerprint
        result['terrain_interpolation'] = 'bilinear_centres_one_sided_halfcell_extension_v1'
        return result

    @classmethod
    def from_files(cls, csv_path, metadata_path, roughness_path=None, manning_n=None):
        csv_path, metadata_path = Path(csv_path), Path(metadata_path)
        metadata = json.loads(metadata_path.read_text(encoding='utf-8-sig'))
        expected = metadata.get('sha256')
        actual = hashlib.sha256(csv_path.read_bytes()).hexdigest()
        if expected is not None and expected != actual:
            raise ValueError('INPUT_DATA_INVALID: terrain.csv SHA-256 mismatch')
        nx, ny = _integer(metadata.get('nx'), 'nx'), _integer(metadata.get('ny'), 'ny')
        if nx < 2 or ny < 2:
            raise ValueError('UNSUPPORTED_DOMAIN: nx/ny must be at least two')
        rows = _read_csv(csv_path, {'cell_id','row','col','x','y','z'})
        if len(rows) != nx*ny:
            raise ValueError('INPUT_DATA_INVALID: terrain grid has missing or extra rows')
        z = np.empty((ny,nx), dtype=float)
        seen = set()
        dx, dy = _positive(metadata.get('dx'),'dx'), _positive(metadata.get('dy'),'dy')
        ox, oy = _finite(metadata.get('x_edge_origin'),'x_edge_origin'), _finite(metadata.get('y_edge_origin'),'y_edge_origin')
        tolerance = float(metadata.get('coordinate_tolerance_m', 1e-7))
        if not np.isfinite(tolerance) or tolerance <= 0 or tolerance > min(dx,dy)*1e-3:
            raise ValueError('INPUT_DATA_INVALID: invalid coordinate tolerance')
        for record in rows:
            cell = _integer(record['cell_id'], 'cell_id')
            r, c = _integer(record['row'],'row'), _integer(record['col'],'col')
            if cell in seen or not (0 <= r < ny and 0 <= c < nx) or cell != r*nx+c:
                raise ValueError('INPUT_DATA_INVALID: duplicate or inconsistent cell index')
            seen.add(cell)
            x, y = _finite(record['x'],'x'), _finite(record['y'],'y')
            if abs(x-(ox+(c+.5)*dx)) > tolerance or abs(y-(oy+(r+.5)*dy)) > tolerance:
                raise ValueError('INPUT_DATA_INVALID: grid centre coordinate mismatch')
            z[r,c] = _finite(record['z'],'z')
        if roughness_path is not None:
            if manning_n is not None:
                raise ValueError('INPUT_DATA_INVALID: choose roughness file or constant, not both')
            roughness_hash=metadata.get('roughness_sha256')
            if roughness_hash is not None and hashlib.sha256(Path(roughness_path).read_bytes()).hexdigest()!=roughness_hash:
                raise ValueError('INPUT_DATA_INVALID: roughness SHA-256 mismatch')
            nrows = _read_csv(roughness_path, {'cell_id','manning_n'})
            if len(nrows) != nx*ny:
                raise ValueError('INPUT_DATA_INVALID: roughness must cover complete grid')
            n = np.empty((ny,nx)); seen = set()
            for record in nrows:
                cell = _integer(record['cell_id'],'cell_id')
                if cell in seen or not 0 <= cell < nx*ny:
                    raise ValueError('INPUT_DATA_INVALID: roughness index mismatch')
                seen.add(cell)
                n.flat[cell] = _positive(record['manning_n'],'manning_n')
        else:
            if manning_n is None:
                raise ValueError('INPUT_DATA_INVALID: explicitly provide roughness file or manning_n')
            n = np.full((ny,nx), _positive(manning_n,'manning_n'))
        return cls(metadata,z,n)

    @classmethod
    def synthetic(cls,nx=6,ny=5,dx=20.,dy=20.,kind='undulating',manning_n=.02):
        nx,ny = _integer(nx,'nx'),_integer(ny,'ny')
        dx,dy = _positive(dx,'dx'),_positive(dy,'dy')
        metadata = dict(schema_version='1.0',grid_id=f'synthetic-{kind}-{nx}x{ny}',
                        terrain_id=f'synthetic-{kind}',terrain_version='1',synthetic=True,
                        crs='LOCAL_ENGINEERING',crs_type='engineering',surface_type='synthetic',
                        axis_order='x_east_y_north',horizontal_unit='m',vertical_unit='m',
                        vertical_datum='synthetic_local_zero',x_edge_origin=0.,y_edge_origin=0.,
                        dx=dx,dy=dy,nx=nx,ny=ny,extent=[0.,nx*dx,0.,ny*dy],
                        row_order='south_to_north',nodata_policy='reject',domain_type='rectangle',
                        assumptions=['Synthetic test fixture, not Minxiong terrain',f'Uniform Manning n={manning_n}'])
        x,y = np.meshgrid((np.arange(nx)+.5)*dx,(np.arange(ny)+.5)*dy)
        if kind=='flat': z=np.zeros_like(x)
        elif kind=='slope': z=10.-.01*x+.005*y
        elif kind=='undulating': z=10.-.005*x+.3*np.sin(2*np.pi*x/(nx*dx))*np.cos(2*np.pi*y/(ny*dy))
        else: raise ValueError('Unknown synthetic terrain kind')
        return cls(metadata,z,np.full_like(z,_positive(manning_n,'manning_n')))

    def query(self,x,y):
        x,y = np.broadcast_arrays(np.asarray(x,dtype=float),np.asarray(y,dtype=float))
        xmin,xmax,ymin,ymax = self.bounds
        if not np.isfinite(x).all() or not np.isfinite(y).all() or np.any((x<xmin)|(x>xmax)|(y<ymin)|(y>ymax)):
            raise ValueError('INPUT_DATA_INVALID: terrain query outside domain/nonfinite')
        # Interior bilinear surface; outer half-cells use the nearest two-centre line.
        px=(x-self.x_origin)/self.dx-.5; py=(y-self.y_origin)/self.dy-.5
        c=np.clip(np.floor(px).astype(int),0,self.nx-2)
        r=np.clip(np.floor(py).astype(int),0,self.ny-2)
        a,b=px-c,py-r
        def interpolate(values):
            ll,lr,ul,ur=values[r,c],values[r,c+1],values[r+1,c],values[r+1,c+1]
            value=(1-a)*(1-b)*ll+a*(1-b)*lr+(1-a)*b*ul+a*b*ur
            zx=((1-b)*(lr-ll)+b*(ur-ul))/self.dx
            zy=((1-a)*(ul-ll)+a*(ur-lr))/self.dy
            return value,zx,zy
        z,zx,zy=interpolate(self.z)
        # Roughness extension uses clamped coordinates to retain positivity near edges.
        a,b=np.clip(a,0,1),np.clip(b,0,1)
        n,_,_=interpolate(self.n)
        return {'z':z,'zx':zx,'zy':zy,'n':n}


def _validate_projected_crs(crs,crs_type):
    if not isinstance(crs,str) or not crs.strip() or crs_type!='projected':
        raise ValueError('INPUT_DATA_INVALID: a declared projected CRS is required')
    try:
        from pyproj import CRS
    except ImportError:
        # Conservative fallback for standard projected systems used in team handoffs.
        normalized=crs.strip().upper()
        known=False
        if normalized.startswith('EPSG:'):
            try: code=int(normalized[5:])
            except ValueError: code=-1
            known=code in {3825,3826,3827,3828,3857} or 32601<=code<=32660 or 32701<=code<=32760
        if not known:
            raise ValueError('INPUT_DATA_INVALID: CRS cannot be verified as projected; install pyproj to validate this CRS')
    else:
        try: projection=CRS.from_user_input(crs)
        except Exception: raise ValueError('INPUT_DATA_INVALID: unparseable CRS') from None
        if not projection.is_projected or not all(abs(axis.unit_conversion_factor-1)<1e-12 for axis in projection.axis_info[:2]):
            raise ValueError('INPUT_DATA_INVALID: CRS must be projected in metre units')


def _finite(value,name):
    if isinstance(value,(bool,np.bool_)): raise ValueError(f'INPUT_DATA_INVALID: invalid {name}')
    try: result=float(value)
    except (TypeError,ValueError): raise ValueError(f'INPUT_DATA_INVALID: invalid {name}') from None
    if not np.isfinite(result): raise ValueError(f'INPUT_DATA_INVALID: nonfinite {name}')
    return result


def _positive(value,name):
    result=_finite(value,name)
    if result<=0: raise ValueError(f'INPUT_DATA_INVALID: {name} must be positive')
    return result


def _integer(value,name):
    result=_finite(value,name)
    if not result.is_integer(): raise ValueError(f'INPUT_DATA_INVALID: {name} must be integer')
    return int(result)


def _read_csv(path,required):
    with Path(path).open(newline='',encoding='utf-8-sig') as stream:
        reader=csv.DictReader(stream)
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f'INPUT_DATA_INVALID: missing columns {sorted(required-set(reader.fieldnames or []))}')
        rows=list(reader)
        if any(None in row for row in rows):
            raise ValueError('INPUT_DATA_INVALID: malformed CSV row')
        return rows
