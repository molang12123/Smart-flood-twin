"""Explicit-unit constant rainfall and deterministic event-aware output times."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Rainfall:
    intensity_mm_hr: float
    duration_s: float

    def __post_init__(self):
        for name in ['intensity_mm_hr','duration_s']:
            if isinstance(getattr(self,name),(bool,np.bool_)):
                raise ValueError(f'INVALID_ARGUMENT: invalid {name}')
            try: value=float(getattr(self,name))
            except (ValueError,TypeError): raise ValueError(f'INVALID_ARGUMENT: invalid {name}') from None
            if not np.isfinite(value) or value<0:
                raise ValueError(f'INVALID_ARGUMENT: {name} must be finite and nonnegative')
            object.__setattr__(self,name,value)

    @property
    def rate_ms(self):
        return self.intensity_mm_hr/3_600_000.

    @classmethod
    def from_duration(cls,intensity_mm_hr,value,unit):
        factors={'s':1.,'min':60.,'h':3600.}
        if unit not in factors: raise ValueError('INVALID_ARGUMENT: duration unit must be s/min/h')
        try: seconds=float(value)*factors[unit]
        except (TypeError,ValueError): raise ValueError('INVALID_ARGUMENT: invalid duration') from None
        return cls(intensity_mm_hr,seconds)

    def rate(self,t):
        t=np.asarray(t,dtype=float)
        if not np.isfinite(t).all(): raise ValueError('INVALID_ARGUMENT: nonfinite time')
        return np.where((t>=0)&(t<self.duration_s),self.rate_ms,0.)

    def integral(self,t0,t1):
        t0,t1=np.broadcast_arrays(np.asarray(t0,dtype=float),np.asarray(t1,dtype=float))
        if not np.isfinite(t0).all() or not np.isfinite(t1).all() or np.any(t1<t0):
            raise ValueError('INVALID_ARGUMENT: invalid integration interval')
        return self.rate_ms*(np.clip(t1,0,self.duration_s)-np.clip(t0,0,self.duration_s))


def output_times(end_s,interval_s,rain_duration_s,requested=None,mode='complete'):
    try: end_s,interval_s,rain_duration_s=map(float,(end_s,interval_s,rain_duration_s))
    except (ValueError,TypeError): raise ValueError('INVALID_ARGUMENT: invalid output schedule') from None
    if not all(np.isfinite([end_s,interval_s,rain_duration_s])) or end_s<=0 or interval_s<=0 or not 0<=rain_duration_s<=end_s:
        raise ValueError('INVALID_ARGUMENT: invalid output schedule bounds')
    if mode not in ['complete','requested']:
        raise ValueError('INVALID_ARGUMENT: output mode must be complete/requested')
    req=np.asarray([] if requested is None else requested,dtype=float)
    if req.ndim!=1 or not np.isfinite(req).all() or np.any((req<0)|(req>end_s)):
        raise ValueError('INVALID_ARGUMENT: output times must be finite and inside [0,T_end]')
    if mode=='requested':
        if len(req)==0: raise ValueError('INVALID_ARGUMENT: requested mode needs at least one time')
        return _deduplicate(req,end_s)
    count=int(np.floor(end_s/interval_s))
    if count>1_000_000: raise ValueError('RESOURCE_LIMIT: output schedule exceeds one million intervals')
    times=_deduplicate(np.concatenate((req,np.arange(count+1)*interval_s,[0.,rain_duration_s,end_s])),end_s)
    if rain_duration_s>0 and not np.any((times>0)&(times<rain_duration_s)):
        times=np.append(times,rain_duration_s/2)
    if rain_duration_s<end_s and not np.any((times>rain_duration_s)&(times<end_s)):
        times=np.append(times,(rain_duration_s+end_s)/2)
    return _deduplicate(times,end_s)


def _deduplicate(values,end_s):
    values=np.sort(np.asarray(values,dtype=float))
    tolerance=8*np.finfo(float).eps*end_s
    if len(values)==0: return values
    result=values[np.concatenate(([True],np.diff(values)>tolerance))]
    # Keep the exact configured end rather than its arithmetic neighbour.
    if abs(result[-1]-end_s)<=tolerance: result[-1]=end_s
    return result
