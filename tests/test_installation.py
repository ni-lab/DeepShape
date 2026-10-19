"""Installation contracts and regressions for the off-site workflow."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'utils'))
from config_paths import expanded_config_text


def workflows_installed():
    return importlib.util.find_spec('selene_sdk') is not None


class InstallationTests(unittest.TestCase):
    def test_fixture_is_complete_and_has_correct_feature_order(self):
        fixture=ROOT/'examples/quickstart'
        manifest=json.loads((fixture/'manifest.json').read_text())
        for name,expected in manifest['files_sha256'].items():
            self.assertEqual(hashlib.sha256((fixture/name).read_bytes()).hexdigest(),expected,name)
        with np.load(fixture/'inputs.npz')as data:
            self.assertEqual(data['encodings'].shape,(16,9,1000))
            self.assertEqual(data['labels'].shape,(16,919))
            self.assertEqual(data['labels'][:,504].sum(),8)
        self.assertEqual((ROOT/'data/distinct_features.txt').read_text().splitlines()[504],'HepG2|MafK|None')

    def test_paths_do_not_depend_on_working_directory(self):
        current=Path.cwd()
        with tempfile.TemporaryDirectory()as directory:
            try:
                os.chdir(directory)
                text=expanded_config_text(ROOT/'config/deeplift_quickstart.yml')
                self.assertIn(str(ROOT/'model/deepshape_no_normalize.py'),text)
                self.assertNotIn('${',text)
            finally:
                os.chdir(current)

    def test_old_placeholder_paths_produce_a_useful_error(self):
        with tempfile.TemporaryDirectory()as directory:
            path=Path(directory)/'bad.yml';path.write_text('path: /ABSOLUTE/PATH/model.py')
            with self.assertRaisesRegex(ValueError,'Replace placeholder paths'):
                expanded_config_text(path)

    @unittest.skipUnless(workflows_installed(),'Install requirements-workflows.txt for genome-reader tests')
    def test_normalized_genome_reader_matches_encoded_fixture(self):
        path=ROOT/'utils/genome_shape_hdf5/genomeshape_normalize.py'
        spec=importlib.util.spec_from_file_location('reader_for_test',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        fixture=ROOT/'examples/quickstart'
        reader=module.GenomeShape(input_path=str(fixture/'genome.fa'),multi_feat_file=str(fixture/'shape.h5'))
        with np.load(fixture/'inputs.npz')as data:
            for index in [0,7,8,15]:
                observed=reader.get_encoding_from_coords('example_%02d'%index,0,1000,'+').T
                np.testing.assert_allclose(observed,data['encodings'][index],rtol=1e-6,atol=1e-7)

    def test_motif_runner_propagates_failure_and_preserves_inputs(self):
        with tempfile.TemporaryDirectory()as directory:
            path=Path(directory);fake=path/'modisco'
            fake.write_text('#!'+sys.executable+'\nraise SystemExit(7)\n');fake.chmod(0o755)
            arrays=np.zeros((2,9,1000),dtype=np.float32)
            for name in ['encodings','attributions']:np.savez(path/(name+'.npz'),arrays)
            output=path/'outputs'
            config=path/'motifs.yml'
            config.write_text('encodings: '+str(path/'encodings.npz')+'\nattributions: '+str(path/'attributions.npz')+'\noutput_dir: '+str(output)+'\nmax_seqlets: 100\n')
            env=dict(os.environ,PATH=str(path)+os.pathsep+os.environ.get('PATH',''))
            result=subprocess.run([sys.executable,str(ROOT/'interpret/run_tfmodisco.py'),str(config)],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            self.assertNotEqual(result.returncode,0)
            self.assertIn(b'CalledProcessError',result.stderr)
            self.assertTrue((output/'seq_onehot.npz').exists())
            self.assertTrue((output/'seq_attribs.npz').exists())

    def test_cli_help(self):
        for script in ['examples/quickstart.py','utils/run_deepshape.py','interpret/run_deeplift.py','interpret/run_tfmodisco.py']:
            result=subprocess.run([sys.executable,str(ROOT/script),'--help'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            self.assertEqual(result.returncode,0,(script,result.stderr.decode()))
            self.assertIn(b'usage:',result.stdout)

if __name__=='__main__':unittest.main()
