import argparse
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/r1_aligned_20261009'))
from plan import BASE,write
from skip_policy import install


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--job-file',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    cli=parser.parse_args()
    sys.path.insert(0,str(BASE/'source'))
    import trainer
    install(trainer,cli.output,write)
    runpy.run_path(str(ROOT/'scripts/r1_aligned_20261009/bootstrap.py'),run_name='__main__')


if __name__=='__main__':main()
