import argparse


def get_asparagus_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=12, help="Number of processes to use.")
    parser.add_argument("--bidsify", action="store_true", help="Restructure dataset in BIDS format.")
    parser.add_argument("--save_dset_metadata", action="store_true", help="Save dataset level metadata.")
    parser.add_argument(
        "--target-size",
        nargs=3,
        type=int,
        default=None,
        metavar=("X", "Y", "Z"),
        help=(
            "Aspect-preserving preprocessing output envelope, e.g. 96 96 96. "
            "Images are uniformly resized and padded, not stretched."
        ),
    )
    parser.add_argument(
        "--target-spacing",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Physical voxel spacing in mm, e.g. --target-spacing 2.5 2.5 2.5.",
    )
    parser.add_argument(
        "--output-size",
        nargs=3,
        type=int,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Fixed output envelope applied after physical resampling, e.g. --output-size 96 96 96.",
    )
    parser.add_argument(
        "--overflow-policy",
        choices=["reject"],
        default="reject",
        help="Action when physically resampled foreground does not fit inside --output-size.",
    )
    parser.add_argument(
        "--allow-legacy-anisotropic-target-size",
        action="store_true",
        help="Explicitly allow legacy fixed-shape resizing for anisotropic medical images.",
    )
    parser.add_argument(
        "--downsample-factor",
        type=float,
        default=None,
        help=(
            "Uniform voxel-grid downsampling factor, e.g. 2.0 halves each spatial dimension. "
            "Output shapes remain variable and proportions are preserved."
        ),
    )
    parser.add_argument(
        "--crop-to-nonzero",
        action="store_true",
        help="Crop background before resampling. Leave disabled when cropping should happen only at train time.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--save_as_tensor",
        action="store_true",
        help="Save processed images as .pt tensors",
    )
    group.add_argument(
        "--save_as_nifti",
        action="store_true",
        help="Save processed images as .nii.gz files",
    )
    return parser


asparagus_parser = get_asparagus_parser()
