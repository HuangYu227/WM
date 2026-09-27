import subprocess
import sys


def test_infer_cli_exposes_sap_modes_without_cuda_imports():
    result = subprocess.run([sys.executable, '-m', 'worldttt', 'infer', '--help'],
                            capture_output=True, text=True, check=True)
    assert 'sap_frozen' in result.stdout
    assert 'sap_online' in result.stdout


def test_evaluator_names_pair_online_with_same_frozen_sap_adapter():
    from worldttt.evaluate import SAP_MODES

    assert SAP_MODES == ('sap_frozen', 'sap_online', 'sap_no_read',
                         'sap_no_commit', 'sap_shuffle_text')
