import argparse
from pathlib import Path
import re
import tempfile
import zipfile
from assignment_lib.runner import ROOT,run_checks,print_result

REQUIRED=['cnn.py','cnn_weights.pt','transformer.py','config.json','format.json']
# The PDF is required for the manual 5 points, but its absence must not block code grading.
# Copy acknowledgements if present; deliberately do not check presence or contents when packing.
OPTIONAL=['cnn_training.json','analysis.pdf','acknowledgement.json']

def pack(directory,student_id,output):
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,40}',student_id):raise ValueError('Use a simple student ID without spaces or paths.')
    directory=Path(directory);output=Path(output)
    if output.exists():raise FileExistsError(f'{output} already exists; choose a new filename or remove your old archive.')
    for name in REQUIRED:
        path=directory/name
        if not path.is_file() or path.is_symlink():raise ValueError(f'Missing regular file: {name}')
    with tempfile.TemporaryDirectory() as tmp:
        report=run_checks(directory,ROOT/'public_profile.json',Path(tmp)/'check.json',interfaces_only=True)
        print_result(report)
        if not report['ok']:raise ValueError('Fix interface errors before packaging.')
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name in REQUIRED+OPTIONAL:
            path=directory/name
            if path.is_file() and not path.is_symlink():archive.write(path,name)
    print('Created',output)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--submission',default='submission');p.add_argument('--student-id',required=True)
    p.add_argument('--output');a=p.parse_args()
    pack(a.submission,a.student_id,a.output or f'{a.student_id}_assignment3.zip')
