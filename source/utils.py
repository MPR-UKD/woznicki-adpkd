import numpy as np
import nibabel as nib
from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform

# The coronal model was trained on data from Groningen in RAS orientation with
# the anterior-posterior voxel axis flipped, i.e. voxel order RPS. Reorienting
# to RPS (instead of RAS + an unrecorded voxel flip) feeds the network the same
# voxel array while keeping the affine consistent with the data.
CORONAL_MODEL_AXCODES = ('R', 'P', 'S')

# Tolerance (mm) when comparing affines; SimpleITK round-trips through float32.
AFFINE_ATOL = 1e-2


def get_orientation(nifti):
    """Determine the acquisition plane from the axis with the fewest voxels"""
    coordinates = nib.aff2axcodes(nifti.affine)
    volume_shapes = nifti.header['dim'][1:4]
    minimal_index = np.argmin(volume_shapes)
    if coordinates[minimal_index] in ['S', 'I']:
        plane = 'ax'
    elif coordinates[minimal_index] in ['A', 'P']:
        plane = 'cor'
    else:
        print('Error, image in sagittal plane - no sagittal model')
        plane = ''
    return plane


def get_task(plane):
    """returns name of task which was different for axial and coronal modes"""
    if plane == 'ax':
        return 'Task002_Kidney'
    else:
        return 'Task003_coronal'


def to_orientation(nifti, axcodes):
    """Reorder voxel axes to `axcodes`, updating data and affine together"""
    transform = ornt_transform(io_orientation(nifti.affine), axcodes2ornt(axcodes))
    return nifti.as_reoriented(transform)


def reorient_for_coronal_model(nifti_path):
    """Reorient a coronal input in place to the voxel order the coronal model expects"""
    nifti = nib.load(nifti_path)
    nib.save(to_orientation(nifti, CORONAL_MODEL_AXCODES), nifti_path)


def compare_geometry(nifti, reference):
    """Return a list of mismatches between the voxel grids of two niftis (empty if identical)"""
    problems = []
    if nifti.shape[:3] != reference.shape[:3]:
        problems.append(f'shape {nifti.shape[:3]} != {reference.shape[:3]}')
    axcodes, ref_axcodes = nib.aff2axcodes(nifti.affine), nib.aff2axcodes(reference.affine)
    if axcodes != ref_axcodes:
        problems.append(f'axcodes {axcodes} != {ref_axcodes}')
    if not np.allclose(nifti.affine, reference.affine, atol=AFFINE_ATOL):
        max_diff = np.abs(nifti.affine - reference.affine).max()
        problems.append(f'affine differs by up to {max_diff:.4g}')
    return problems


def restore_input_geometry(seg_path, image_path):
    """Map a predicted segmentation back onto the voxel grid of the original image, in place.

    The prediction carries the geometry of the (possibly reoriented) network input,
    so reorienting it to the original image's axcodes is an exact inverse.
    Raises ValueError if the result does not match the original image's grid.
    """
    image = nib.load(image_path)
    seg = to_orientation(nib.load(seg_path), nib.aff2axcodes(image.affine))
    problems = compare_geometry(seg, image)
    if problems:
        raise ValueError(f'segmentation does not match input geometry: {"; ".join(problems)}')

    header = image.header.copy()
    header.set_data_dtype(np.uint8)
    header.set_slope_inter(1, 0)
    data = np.asanyarray(seg.dataobj).astype(np.uint8)
    nib.save(nib.Nifti1Image(data, image.affine, header), seg_path)
