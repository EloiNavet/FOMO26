def crop_to_box(array, bbox):
    """
    Crop using a bounding box with exclusive upper indices.

    ``get_bbox_for_label`` returns ``[xmin, xmax, ymin, ymax, ...]`` where
    each maximum already follows Python's exclusive slice convention.
    """
    if len(bbox) > 5:
        bbox_slices = (
            slice(bbox[0], bbox[1]),
            slice(bbox[2], bbox[3]),
            slice(bbox[4], bbox[5]),
        )
    else:
        bbox_slices = (slice(bbox[0], bbox[1]), slice(bbox[2], bbox[3]))
    return array[bbox_slices]
