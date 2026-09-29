"""Check that every mask in a job's output dir matches the geometry of its input image.

Usage: python check_orientation.py <input.nii.gz> <output_dir>
Exits 1 if any mask's shape, axcodes or affine differ from the input image.
"""

import sys
from pathlib import Path

import nibabel as nib

from utils import compare_geometry

if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    image = nib.load(sys.argv[1])
    masks = sorted(Path(sys.argv[2]).glob('*.nii.gz'))
    if not masks:
        sys.exit(f'no .nii.gz masks found in {sys.argv[2]}')

    print(f'input: shape={image.shape[:3]} axcodes={nib.aff2axcodes(image.affine)}')
    failed = False
    for mask_path in masks:
        problems = compare_geometry(nib.load(mask_path), image)
        print(f'{mask_path.name}: {"OK" if not problems else "; ".join(problems)}')
        failed = failed or bool(problems)
    sys.exit(1 if failed else 0)
