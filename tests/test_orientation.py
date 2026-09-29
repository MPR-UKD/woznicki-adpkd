"""Orientation round-trip tests for the pre-/post-processing around nnU-Net.

A stand-in "prediction" is derived voxel-wise from the network input and written
with SimpleITK the way nnU-Net does, so the tests exercise the real geometry
hand-off without needing a model or GPU.
"""

import nibabel as nib
import numpy as np
import pytest
import SimpleITK as sitk
from nibabel.orientations import axcodes2ornt, ornt_transform

from postprocess_masks import is_right_kidney
from utils import (
    CORONAL_MODEL_AXCODES,
    compare_geometry,
    reorient_for_coronal_model,
    restore_input_geometry,
    to_orientation,
)

# Voxel counts per anatomical direction; A/P is thinnest, as for a coronal scan.
SHAPE_BY_DIRECTION = {'LR': 12, 'AP': 5, 'SI': 9}
SPACING_BY_DIRECTION = {'LR': 0.8, 'AP': 6.0, 'SI': 1.5}

CORONAL_LAYOUTS = ['LSP', 'RSA', 'LPS', 'PRS', 'LIP', 'ASL']


def _direction(code):
    return next(key for key in SHAPE_BY_DIRECTION if code in key)


def make_image(axcodes, oblique=False):
    """Build a nifti with voxel order `axcodes` and fully asymmetric intensities."""
    ras_codes = ('R', 'A', 'S')
    shape = tuple(SHAPE_BY_DIRECTION[_direction(c)] for c in ras_codes)
    affine = np.diag([SPACING_BY_DIRECTION[_direction(c)] for c in ras_codes] + [1.0])
    affine[:3, 3] = [-40.0, 25.0, 110.0]
    if oblique:
        angle = np.deg2rad(12)
        rotation = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ])
        affine[:3, :3] = rotation @ affine[:3, :3]
    # A random permutation so the labels below change under a flip of any axis.
    rng = np.random.default_rng(seed=0)
    data = rng.permutation(np.prod(shape)).astype(np.int16).reshape(shape)
    ras_image = nib.Nifti1Image(data, affine)
    return to_orientation(ras_image, tuple(axcodes))


def labels_from_intensities(data):
    """Deterministic, asymmetric stand-in for a segmentation of an image."""
    return (np.asarray(data) % 2 == 0).astype(np.uint8)


def predict_like_nnunet(input_path, output_path):
    """Write a label map for `input_path` carrying its geometry, as nnU-Net's export does."""
    image = sitk.ReadImage(str(input_path))
    seg = sitk.GetImageFromArray(labels_from_intensities(sitk.GetArrayFromImage(image)))
    seg.SetSpacing(image.GetSpacing())
    seg.SetOrigin(image.GetOrigin())
    seg.SetDirection(image.GetDirection())
    sitk.WriteImage(seg, str(output_path))


def run_pipeline(image, tmp_path, coronal):
    """Run pre-processing, stand-in inference and post-processing; return the restored mask."""
    image_path = tmp_path / 'image.nii.gz'
    network_input_path = tmp_path / '1_0000.nii.gz'
    seg_path = tmp_path / 'seg.nii.gz'
    nib.save(image, image_path)
    nib.save(image, network_input_path)
    if coronal:
        reorient_for_coronal_model(network_input_path)
    predict_like_nnunet(network_input_path, seg_path)
    restore_input_geometry(seg_path, image_path)
    return nib.load(seg_path)


@pytest.mark.unit
class TestCoronalPreprocessing:
    """The coronal network input keeps its voxels but gains a consistent affine."""

    @pytest.mark.parametrize('axcodes', CORONAL_LAYOUTS)
    def test_voxels_match_legacy_ras_plus_flip(self, axcodes, tmp_path):
        """RPS reorientation yields the same voxel array as the old RAS reorient + axis-1 flip."""
        image = make_image(axcodes)
        path = tmp_path / 'input.nii.gz'
        nib.save(image, path)

        reorient_for_coronal_model(path)

        legacy_ras = image.as_reoriented(
            ornt_transform(nib.orientations.io_orientation(image.affine), axcodes2ornt('RAS'))
        )
        legacy_voxels = np.flip(np.asanyarray(legacy_ras.dataobj), axis=1)
        np.testing.assert_array_equal(np.asanyarray(nib.load(path).dataobj), legacy_voxels)

    @pytest.mark.parametrize('axcodes', CORONAL_LAYOUTS)
    def test_affine_describes_rps_voxel_order(self, axcodes, tmp_path):
        """The reoriented file is labelled RPS, i.e. its affine records the flip."""
        path = tmp_path / 'input.nii.gz'
        nib.save(make_image(axcodes), path)

        reorient_for_coronal_model(path)

        assert nib.aff2axcodes(nib.load(path).affine) == CORONAL_MODEL_AXCODES


@pytest.mark.unit
class TestRestoreInputGeometry:
    """The returned mask lies on the original image's voxel grid."""

    @pytest.mark.parametrize('oblique', [False, True], ids=['straight', 'oblique'])
    @pytest.mark.parametrize('axcodes', CORONAL_LAYOUTS)
    def test_coronal_round_trip_matches_input(self, axcodes, oblique, tmp_path):
        """Coronal inputs of any voxel layout come back aligned voxel-for-voxel."""
        image = make_image(axcodes, oblique=oblique)

        seg = run_pipeline(image, tmp_path, coronal=True)

        assert compare_geometry(seg, image) == []
        np.testing.assert_array_equal(
            np.asanyarray(seg.dataobj), labels_from_intensities(image.dataobj)
        )

    @pytest.mark.parametrize('axcodes', ['LPS', 'RAS', 'LAI'])
    def test_axial_round_trip_matches_input(self, axcodes, tmp_path):
        """Axial inputs pass through unchanged."""
        image = make_image(axcodes, oblique=True)

        seg = run_pipeline(image, tmp_path, coronal=False)

        assert compare_geometry(seg, image) == []
        np.testing.assert_array_equal(
            np.asanyarray(seg.dataobj), labels_from_intensities(image.dataobj)
        )

    def test_output_is_uint8_labels(self, tmp_path):
        """The restored mask is stored as unscaled uint8."""
        seg = run_pipeline(make_image('LSP'), tmp_path, coronal=True)

        assert seg.get_data_dtype() == np.uint8
        assert seg.dataobj.slope == 1 and seg.dataobj.inter == 0

    def test_mismatched_grid_raises(self, tmp_path):
        """A prediction on a different grid is rejected instead of saved."""
        image_path, seg_path = tmp_path / 'image.nii.gz', tmp_path / 'seg.nii.gz'
        nib.save(make_image('LPS'), image_path)
        nib.save(nib.Nifti1Image(np.zeros((3, 3, 3), np.uint8), np.eye(4)), seg_path)

        with pytest.raises(ValueError, match='does not match input geometry'):
            restore_input_geometry(seg_path, image_path)


@pytest.mark.unit
class TestIsRightKidney:
    """Left/right assignment uses the L/R axis wherever it sits in the voxel order."""

    @pytest.mark.parametrize(
        ('axcodes', 'center', 'expected_right'),
        [
            (('R', 'A', 'S'), (10, 2, 2), True),
            (('L', 'S', 'P'), (10, 2, 2), False),
            (('P', 'R', 'S'), (2, 10, 2), True),
            (('A', 'S', 'L'), (2, 2, 10), False),
        ],
    )
    def test_uses_lr_axis(self, axcodes, center, expected_right):
        """A component on the high-index side of the L/R axis is right iff that axis points R."""
        mask = to_orientation(nib.Nifti1Image(np.zeros((12, 12, 12), np.uint8), np.eye(4)), axcodes)

        assert is_right_kidney(mask, center) == expected_right
