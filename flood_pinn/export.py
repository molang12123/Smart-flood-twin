"""Streaming CSV/JSON export of fixed-grid simulations and volume accounting."""
from contextlib import contextmanager
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import uuid
import numpy as np
from .exchange import (build_faces, cell_sides, evaluate_points,
                       physical_points, integrate_exchange)


def _default(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    raise TypeError(f'Not JSON serializable: {type(value).__name__}')


def _json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False, default=_default), encoding='utf-8')


def _writer(stack, path, names):
    stream = stack.enter_context(Path(path).open('w', newline='', encoding='utf-8'))
    writer = csv.DictWriter(stream, fieldnames=names)
    writer.writeheader()
    return writer



def _plotting():
    from .platform_info import prepare_runtime_cache
    prepare_runtime_cache()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def _grid_plot(terrain, h, u, v, t, directory, frame):
    plt=_plotting()
    extent = [terrain.bounds[0], terrain.bounds[1], terrain.bounds[2], terrain.bounds[3]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im = axes[0].imshow(h.reshape(terrain.ny, terrain.nx), origin='lower', extent=extent, cmap='Blues')
    fig.colorbar(im, ax=axes[0], label='Depth [m]')
    step = max(1, len(terrain.x)//400)
    if np.any(np.hypot(u,v)>0):
        arrow_scale=float(np.hypot(u,v).max())/(.8*min(terrain.dx,terrain.dy))
        axes[1].quiver(terrain.x[::step], terrain.y[::step], u[::step], v[::step],
                       angles='xy',scale_units='xy',scale=arrow_scale)
    else:
        axes[1].text(.5,.5,'Zero velocity',ha='center',transform=axes[1].transAxes)
        axes[1].set_xlim(extent[0],extent[1]);axes[1].set_ylim(extent[2],extent[3])
    axes[0].set_title(f'Depth t={t:g}s'); axes[1].set_title('Velocity [m/s]')
    for ax in axes:
        ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.set_aspect('equal')
    fig.tight_layout(); fig.savefig(directory / f'fields_{frame:04d}.png', dpi=100); plt.close(fig)



@contextmanager
def _claim_export_directory(directory):
    """OS-exclusive cooperative claim; never remove another owner's lock."""
    lock=directory/'.export.lock'
    try:
        descriptor=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    except FileExistsError as error:
        raise FileExistsError('Output directory is already claimed by an export') from error
    identity=os.fstat(descriptor)
    try:
        # Recheck after the atomic claim: another export may have completed
        # between our caller's initial inspection and lock acquisition.
        if any(entry.name!='.export.lock' for entry in directory.iterdir()):
            raise FileExistsError('output directory must be empty; select a fresh run directory')
        yield
    finally:
        os.close(descriptor)
        try:
            current=lock.lstat()
            if (current.st_dev,current.st_ino)==(identity.st_dev,identity.st_ino): lock.unlink()
        except FileNotFoundError:
            pass


def export_simulation(terrain, evaluate, times, rainfall, out_dir, *, run_metadata=None,
                      dry_depth=1e-4, flood_depth=.01, batch_size=4096,
                      time_substeps=2, make_plots=True):
    """Export raw-field flux accounting and display fields, bounded by one frame.

    Existing results are protected: use a new output directory for every run.
    Outputs are staged then published after success; failure is recorded separately.
    This routine does not train or certify a model, and never alters the input grid.
    """
    from contextlib import ExitStack
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or (times < 0).any() or (np.diff(times) <= 0).any():
        raise ValueError('times must be a nonempty strictly increasing finite nonnegative sequence')
    if not np.isfinite([dry_depth, flood_depth]).all() or dry_depth < 0 or flood_depth <= dry_depth:
        raise ValueError('require 0 <= dry_depth < flood_depth')
    meta = dict(run_metadata or {})
    # Check metadata before running inference, rather than silently coercing invalid numbers.
    json.dumps(meta, allow_nan=False, default=_default)
    end=meta.get('end_s',float(times[-1]))
    if isinstance(end,bool) or not isinstance(end,(int,float,np.number)) or not np.isfinite(end) or end < times[-1]:
        raise ValueError('declared end_s must be finite and not precede the final frame')
    directory = Path(out_dir)
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise FileExistsError('output directory must be empty; select a fresh run directory')
    directory.mkdir(parents=True, exist_ok=True)
    with _claim_export_directory(directory):
        staging = Path(tempfile.mkdtemp(prefix='.export-', dir=directory))
        published=[]
        try:
            summary = _export(terrain, evaluate, times, rainfall, staging, meta,
                              dry_depth, flood_depth, batch_size, time_substeps, make_plots)
            # Publish the completion manifest last. Atomic hard-link creation
            # refuses destination collisions even from a noncooperating writer.
            staged_files = sorted(staging.iterdir(),key=lambda file:(file.name=='manifest.json',file.name))
            if any((directory/file.name).exists() or (directory/file.name).is_symlink() for file in staged_files):
                raise FileExistsError('Output path appeared during export; preserving existing files')
            for file in staged_files:
                target=directory/file.name
                os.link(file,target)
                stat=target.lstat();published.append((target,stat.st_dev,stat.st_ino))
                file.unlink()
            staging.rmdir()
            return summary
        except Exception as error:
            # Roll back only files this export created, preserving replacements.
            for target,device,inode in reversed(published):
                try:
                    stat=target.lstat()
                    if (stat.st_dev,stat.st_ino)==(device,inode): target.unlink()
                except FileNotFoundError:
                    pass
            shutil.rmtree(staging, ignore_errors=True)
            failure={'status':'failed','error_type':type(error).__name__,'message':str(error)}
            failure_path=directory/'simulation_failure.json'
            while True:
                try:
                    with failure_path.open('x',encoding='utf-8') as stream:
                        json.dump(failure,stream,ensure_ascii=False,indent=2,allow_nan=False)
                    break
                except FileExistsError:
                    failure_path=directory/f'simulation_failure_{uuid.uuid4().hex}.json'
            raise


def _export(terrain, evaluate, times, rainfall, directory, meta, dry_depth,
            flood_depth, batch_size, time_substeps, make_plots):
    from contextlib import ExitStack
    faces = build_faces(terrain); sides = cell_sides(terrain)
    count = terrain.nx*terrain.ny; area = float(terrain.dx*terrain.dy)
    z = np.asarray(terrain.z).ravel(); ids = np.arange(count)
    run_id = str(meta.get('run_id', uuid.uuid4().hex))
    summary = {'schema_version': '1.0', 'run_id': run_id, 'status': 'complete',
               'model_id': meta.get('model_id'), 'grid_id': terrain.metadata.get('grid_id'),
               'terrain_fingerprint': terrain.fingerprint,
               'rainfall_mm_hr': rainfall.intensity_mm_hr, 'rain_duration_s': rainfall.duration_s,
               'total_simulation_time_s': meta.get('end_s', float(times[-1])),
               'output_times_s': times.tolist(), 'coverage': meta.get('coverage', 'requested_interval'),
               'peak_definition': 'sampled output times and cell centers, not continuous extrema',
               'reference_validation_status': meta.get('reference_validation_status', 'not_completed'),
               'validation_status': meta.get('validation_status', 'not_evaluated'),
               'dry_depth_m': dry_depth, 'flood_depth_m': flood_depth,
               'assumptions': ['No infiltration or drainage model',
                               'Primary exchange uses raw PINN h/u/v, not display-masked velocities',
                               'Cell storage uses midpoint h*area; discrete conservation is audited, not enforced'],
               'max_depth_m': -1., 'max_speed_m_s': -1., 'max_raw_speed_m_s': -1.,
               'max_flooded_area_m2': -1., 'maximum_dry_raw_speed_m_s': 0.,
               'max_cell_balance_error_m3': 0., 'max_global_balance_error_m3': 0.}
    grid = dict(terrain.to_metadata())
    grid.update({'schema_version': '1.0', 'run_id': run_id, 'terrain_fingerprint': terrain.fingerprint,
                 'field_units': {'x':'m','y':'m','z':'m','h':'m','water_surface':'m','t':'s',
                                 'u':'m/s','v':'m/s','speed':'m/s','direction':'degree',
                                 'qx':'m2/s','qy':'m2/s','Q':'m3/s','V':'m3'},
                 'sort_order': ['t','cell_id'], 'direction_definition': '0 east (+x), 90 north (+y), counterclockwise [0,360)',
                 'undefined_direction': 'CSV empty / JSON null', 'dry_depth_m': dry_depth,
                 'flood_depth_m': flood_depth, 'storage_quadrature': 'cell center h * dx * dy',
                 'face_quadrature': 'two-point Gauss on face; piecewise-linear composite time integration',
                 'time_integration_error': 'fine/coarse refinement difference estimate; not guaranteed bound',
                 'face_id_rule': 'x: row*(nx+1)+k; y: ny*(nx+1)+j*nx+col',
                 'face_normal_rule': 'x faces +x, y faces +y; exterior cell null',
                 'ue_conversion': {'units': 'CSV stays in SI; UE member converts to centimeters',
                                   'formula': 'p_UE_cm=100*B*(p_GIS_m-anchor); velocity_UE=100*B*(u,v,0)',
                                   'anchor': meta.get('ue_anchor'), 'axis_map': meta.get('ue_axis_map'),
                                   'status': 'anchor and axis mapping require team confirmation'}})
    _json(directory/'grid_metadata.json', grid)
    state_names = ['run_id','frame_id','cell_id','row','col','x','y','t','z','h','water_surface',
                   'u','v','speed','direction','flooded','is_dry','qx','qy','u_raw','v_raw']
    face_names = ['run_id','face_id','axis','minus_cell_id','plus_cell_id','boundary_tag','face_x','face_y',
                  'normal_x','normal_y','face_length_m','t_start','t_end','Q_start_m3s','Q_end_m3s',
                  'V_forward_m3','V_reverse_m3','V_signed_m3','integration_method','integration_error_est_m3']
    cell_names = ['run_id','cell_id','t_start','t_end','area_m2']
    for side in sides:
        cell_names.extend(f'{side}_{key}' for key in ['Q_in_end_m3s','Q_out_end_m3s','V_in_m3','V_out_m3'])
    cell_names += ['V_rain_m3','V_infiltration_m3','V_drain_m3','V_start_m3','V_end_m3',
                   'delta_storage_m3','V_net_in_m3','balance_error_m3']
    water_names = ['run_id','t','V_initial_m3','V_storage_m3','V_rain_m3','V_in_m3','V_out_m3',
                   'V_infiltration_m3','V_drain_m3','balance_error_m3','relative_error',
                   'relative_denominator_m3','integration_error_est_m3']
    first_storage = None; previous_h = None
    cumulative_in = cumulative_out = cumulative_error_est = 0.
    water_rows = []
    selected = {0, len(times)//2, len(times)-1}
    with ExitStack() as stack:
        state_csv = _writer(stack, directory/'simulation_results.csv', state_names)
        face_csv = _writer(stack, directory/'flow_exchange.csv', face_names)
        cell_csv = _writer(stack, directory/'cell_water_balance.csv', cell_names)
        water_csv = _writer(stack, directory/'water_balance.csv', water_names)
        for frame, t in enumerate(times):
            field = evaluate_points(evaluate, physical_points(terrain.x, terrain.y, t, rainfall), batch_size)
            h, ur, vr = field.T
            raw_speed = np.hypot(ur, vr)
            dry = h <= dry_depth
            u = np.where(dry, 0., ur); v = np.where(dry, 0., vr); speed = np.hypot(u,v)
            surface = z+h; qx=h*u; qy=h*v
            if not all(np.isfinite(a).all() for a in [surface, raw_speed, speed, qx, qy]):
                raise FloatingPointError('NONFINITE_PREDICTION: derived field')
            direction = np.mod(np.degrees(np.arctan2(v,u)), 360)
            flooded = h >= flood_depth
            for c in ids:
                state_csv.writerow(dict(zip(state_names, [run_id,frame,c,c//terrain.nx,c%terrain.nx,
                    terrain.x[c],terrain.y[c],t,z[c],h[c],surface[c],u[c],v[c],speed[c],
                    '' if dry[c] or speed[c] <= 1e-10 else direction[c],int(flooded[c]),int(dry[c]),
                    qx[c],qy[c],ur[c],vr[c]])))
            for value, key in [(h, 'depth'), (speed, 'speed'), (raw_speed, 'raw_speed')]:
                maximum = float(value.max())
                metric = 'max_depth_m' if key == 'depth' else f'max_{key}_m_s'
                if maximum > summary[metric]:
                    summary[metric] = maximum
                    summary[f'max_{key}_time_s'] = float(t)
                    summary[f'max_{key}_cell_id'] = int(value.argmax())
            flooded_area = float(flooded.sum()*area)
            if flooded_area > summary['max_flooded_area_m2']:
                summary['max_flooded_area_m2'] = flooded_area
                summary['max_flooded_area_time_s'] = float(t)
            if dry.any():
                summary['maximum_dry_raw_speed_m_s'] = max(summary['maximum_dry_raw_speed_m_s'],float(raw_speed[dry].max()))
            storage = float(h.sum()*area)
            if first_storage is None: first_storage = storage
            if previous_h is not None:
                t0 = float(times[frame-1])
                exchange = integrate_exchange(terrain, evaluate, rainfall, t0, float(t), batch_size=batch_size,
                                              time_substeps=time_substeps, faces=faces)
                for f in faces.ids:
                    face_csv.writerow(dict(zip(face_names, [run_id,f,faces.axis[f],
                        None if faces.minus[f]<0 else int(faces.minus[f]),
                        None if faces.plus[f]<0 else int(faces.plus[f]), faces.boundary[f],faces.x[f],faces.y[f],
                        faces.nx[f],faces.ny[f],faces.length[f],t0,t,exchange['Q_start'][f],exchange['Q_end'][f],
                        exchange['V_forward'][f],exchange['V_reverse'][f],exchange['V_signed'][f],
                        'gauss2_face_linear_time_rain_event_split',exchange['error_est'][f]])))
                side_data = {}; vin = np.zeros(count); vout = np.zeros(count)
                for side, (indices, sign) in sides.items():
                    q = sign*exchange['Q_end'][indices]
                    vp = exchange['V_forward'][indices]; vn = exchange['V_reverse'][indices]
                    out, incoming = (vp,vn) if sign == 1 else (vn,vp)
                    side_data[side] = (np.maximum(-q,0),np.maximum(q,0),incoming,out)
                    vin += incoming; vout += out
                rain_cell = float(rainfall.integral(t0,float(t)))*area
                before = previous_h*area; after=h*area
                balance = after-before-(rain_cell+vin-vout)
                if not np.isfinite(balance).all(): raise FloatingPointError('nonfinite cell water balance')
                summary['max_cell_balance_error_m3'] = max(summary['max_cell_balance_error_m3'],float(np.abs(balance).max()))
                for c in ids:
                    row = {'run_id':run_id,'cell_id':c,'t_start':t0,'t_end':t,'area_m2':area}
                    for side, arrays in side_data.items():
                        for key,value in zip(['Q_in_end_m3s','Q_out_end_m3s','V_in_m3','V_out_m3'],arrays):
                            row[f'{side}_{key}']=value[c]
                    row.update(V_rain_m3=rain_cell,V_infiltration_m3=0.,V_drain_m3=0.,V_start_m3=before[c],
                               V_end_m3=after[c],delta_storage_m3=after[c]-before[c],V_net_in_m3=vin[c]-vout[c],
                               balance_error_m3=balance[c])
                    cell_csv.writerow(row)
                boundary = faces.boundary != ''
                positive_out = np.isin(faces.boundary,['east','north'])
                forward = exchange['V_forward']; reverse=exchange['V_reverse']
                cumulative_out += float(np.where(positive_out,forward,reverse)[boundary].sum())
                cumulative_in += float(np.where(positive_out,reverse,forward)[boundary].sum())
                cumulative_error_est += float(exchange['error_est'][boundary].sum())
            rain_volume = float(rainfall.integral(float(times[0]),float(t)))*count*area
            balance_global = storage-first_storage-rain_volume-cumulative_in+cumulative_out
            denominator=max(abs(first_storage)+rain_volume+cumulative_in,1e-12)
            row = dict(zip(water_names,[run_id,t,first_storage,storage,rain_volume,cumulative_in,cumulative_out,0.,0.,
                                        balance_global,balance_global/denominator,denominator,cumulative_error_est]))
            water_csv.writerow(row); water_rows.append(row)
            summary['max_global_balance_error_m3'] = max(summary['max_global_balance_error_m3'],abs(balance_global))
            if make_plots and frame in selected: _grid_plot(terrain,h,u,v,t,directory,frame)
            previous_h=h.copy()
    summary['total_rain_depth_m'] = float(rainfall.integral(0,float(summary['total_simulation_time_s'])))
    summary['total_rain_volume_m3'] = summary['total_rain_depth_m']*count*area
    summary['rain_volume_through_last_frame_m3'] = float(rainfall.integral(0,float(times[-1])))*count*area
    summary['accounted_rain_volume_m3'] = water_rows[-1]['V_rain_m3']
    summary['water_balance'] = {'interval_start_s':float(times[0]),'interval_end_s':float(times[-1]),
                               'initial_storage_m3':first_storage,'final_storage_m3':water_rows[-1]['V_storage_m3'],
                               'rain_volume_m3':water_rows[-1]['V_rain_m3'],'boundary_in_m3':cumulative_in,
                               'boundary_out_m3':cumulative_out,'error_m3':water_rows[-1]['balance_error_m3'],
                               'relative_error':water_rows[-1]['relative_error'],
                               'integration_error_est_m3':cumulative_error_est,
                               'status':'measured_not_certified'}
    summary['num_cells']=count; summary['num_frames']=len(times); summary['num_faces']=len(faces.ids)
    if make_plots: _summary_plots(terrain,water_rows,directory)
    _json(directory/'simulation_summary.json', summary)
    row_counts={'simulation_results.csv':count*len(times),
                'flow_exchange.csv':len(faces.ids)*(len(times)-1),
                'cell_water_balance.csv':count*(len(times)-1),
                'water_balance.csv':len(times)}
    refresh_manifest(directory, row_counts=row_counts, run_id=run_id, grid_id=summary['grid_id'])
    return summary



def refresh_manifest(out_dir, *, row_counts=None, run_id=None, grid_id=None):
    """Refresh hashes after summary/benchmark additions; hash large files in chunks.

    Unchanged CSV files retain previous known row counts. New or changed CSVs
    without an explicit count are counted with the CSV reader, one row at a time.
    """
    directory=Path(out_dir)
    manifest_path=directory/'manifest.json'
    previous={}
    if manifest_path.exists():
        previous=json.loads(manifest_path.read_text(encoding='utf-8'))
    old={item['path']:item for item in previous.get('files',[])}
    counts=dict(row_counts or {})
    files=[]
    for path in sorted(directory.iterdir()):
        if path.name in ('manifest.json','.export.lock') or path.is_dir(): continue
        if path.is_symlink(): raise ValueError('Refusing to hash a symbolic-link artifact')
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''): digest.update(chunk)
        sha=digest.hexdigest(); count=counts.get(path.name)
        if path.suffix=='.csv' and count is None:
            item=old.get(path.name,{})
            if item.get('sha256')==sha and item.get('row_count') is not None:
                count=item['row_count']
            else:
                with path.open(newline='',encoding='utf-8') as stream:
                    reader=csv.reader(stream); next(reader,None)
                    count=sum(1 for _ in reader)
        files.append({'path':path.name,'sha256':sha,'bytes':path.stat().st_size,'row_count':count})
    result={'schema_version':'1.0','run_id':run_id or previous.get('run_id'),
            'grid_id':grid_id or previous.get('grid_id'),'status':'complete','files':files}
    _json(manifest_path,result)
    return result


def _summary_plots(terrain, rows, directory):
    plt=_plotting()
    fig,ax=plt.subplots(figsize=(6,4))
    im=ax.imshow(terrain.z,origin='lower',extent=terrain.bounds,cmap='terrain')
    ax.set(xlabel='x [m]',ylabel='y [m]',title='Terrain elevation [m]'); fig.colorbar(im,ax=ax)
    fig.tight_layout();fig.savefig(directory/'terrain.png',dpi=100);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(7,6),sharex=True)
    t=[r['t'] for r in rows]
    for key,label in [('V_storage_m3','Storage'),('V_rain_m3','Rain'),('V_in_m3','Boundary in'),('V_out_m3','Boundary out')]:
        axes[0].plot(t,[r[key] for r in rows],label=label)
    axes[0].set_ylabel('Volume [m³]');axes[0].legend()
    axes[1].plot(t,[r['balance_error_m3'] for r in rows]);axes[1].set(xlabel='t [s]',ylabel='Balance error [m³]')
    fig.tight_layout();fig.savefig(directory/'water_balance.png',dpi=100);plt.close(fig)
