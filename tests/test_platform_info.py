import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from flood_pinn.platform_info import prepare_runtime_cache, process_memory_info
import flood_pinn.platform_info as platform_info


class PlatformInfoTests(unittest.TestCase):
    def test_chinese_bom_config_and_json_do_not_depend_on_windows_locale(self):
        from flood_pinn.__main__ import load_config
        from flood_pinn.contracts import write_json
        root=Path(__file__).resolve().parents[1]
        cfg=json.loads((root/'configs/fixed_terrain.json').read_text(encoding='utf-8'))
        for key in ('csv','metadata','roughness'):
            cfg['terrain'][key]=str((root/'configs'/cfg['terrain'][key]).resolve())
        cfg['description']='固定地形：民雄資料介面'
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'中文配置.json'
            path.write_text(json.dumps(cfg,ensure_ascii=False),encoding='utf-8-sig')
            read_text=Path.read_text
            write_text=Path.write_text
            def explicit_read(p,*args,**kwargs):
                self.assertIn('encoding',kwargs)
                return read_text(p,*args,**kwargs)
            def explicit_write(p,*args,**kwargs):
                self.assertEqual(kwargs.get('encoding'),'utf-8')
                return write_text(p,*args,**kwargs)
            with patch.object(Path,'read_text',explicit_read), patch.object(Path,'write_text',explicit_write):
                grid,loaded=load_config(path)
                write_json(Path(d)/'中文成果.json',{'说明':loaded['description']})
            self.assertEqual(loaded['description'],cfg['description'])
            self.assertEqual(json.loads((Path(d)/'中文成果.json').read_bytes().decode('utf-8'))['说明'],cfg['description'])

    def test_caches_use_os_temp_and_preserve_explicit_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict('os.environ', {}, clear=True), patch(
                    'flood_pinn.platform_info.tempfile.gettempdir', return_value=directory):
                paths = prepare_runtime_cache()
                for path in paths.values():
                    self.assertTrue(Path(path).is_dir())
                    self.assertTrue(Path(path).is_relative_to(Path(directory)))
                custom = str(Path(directory) / 'explicit')
                with patch.dict('os.environ', {'MPLCONFIGDIR': custom}):
                    self.assertEqual(prepare_runtime_cache()['MPLCONFIGDIR'], custom)

    def test_windows_does_not_import_resource_or_claim_a_measurement(self):
        with patch('flood_pinn.platform_info.sys.platform', 'win32'), patch(
                'flood_pinn.platform_info.importlib.import_module') as importer:
            result = process_memory_info()
        importer.assert_not_called()
        self.assertIsNone(result['peak_process_rss_kib'])
        self.assertEqual(result['rss_measurement_status'], 'unavailable')
        json.dumps(result, allow_nan=False)

    def test_linux_and_macos_units(self):
        resource = SimpleNamespace(RUSAGE_SELF=0, getrusage=lambda _: SimpleNamespace(ru_maxrss=2048))
        with patch.object(platform_info.importlib, 'import_module', return_value=resource):
            with patch.object(platform_info.sys, 'platform', 'linux'):
                self.assertEqual(process_memory_info()['peak_process_rss_kib'], 2048)
            with patch.object(platform_info.sys, 'platform', 'darwin'):
                self.assertEqual(process_memory_info()['peak_process_rss_kib'], 2)

    def test_missing_resource_is_optional(self):
        with patch('flood_pinn.platform_info.sys.platform', 'linux'), patch(
                'flood_pinn.platform_info.importlib.import_module', side_effect=ImportError):
            self.assertIsNone(process_memory_info()['peak_process_rss_kib'])


if __name__ == '__main__':
    unittest.main()
