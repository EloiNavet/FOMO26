import inspect
import os
from gardening_tools.functional.paths.write import save_prediction_from_logits
from lightning.pytorch.callbacks import BasePredictionWriter

_NIFTI_SUFFIX = ".nii.gz"


def _write_prediction(logits, output_path, properties) -> None:
    """Write to exactly ``output_path`` on either gardening_tools writer API.

    The two versions disagree about whether the caller supplies the extension, and the older one
    fails silently rather than loudly:

      0.3.5  save_prediction_from_logits(logits, outpath, properties)   -> writes outpath
      0.3.2  save_prediction_from_logits(logits, outpath, properties,
                                         save_format="nii.gz")          -> writes outpath + ".nii.gz"

    The release runtime is pinned to 0.3.2 -- the version that trained the packaged checkpoints --
    so passing a path that already ends in ``.nii.gz`` produced ``<case>.nii.gz.nii.gz``. Nothing
    raised: the callback returned normally and the file the caller asked for was simply not there.

    Dispatching on the signature keeps one call site correct on both versions rather than encoding
    a guess about which is installed. ``finetuning/container/fomo_ensemble_predict.py`` already
    does this for the container entrypoint; this callback is the other writer that ships, and it
    did not.
    """
    if "save_format" in inspect.signature(save_prediction_from_logits).parameters:
        text = str(output_path)
        if not text.endswith(_NIFTI_SUFFIX):
            raise ValueError(f"Expected a {_NIFTI_SUFFIX} output path, got {output_path!r}.")
        save_prediction_from_logits(
            logits,
            text[: -len(_NIFTI_SUFFIX)],
            properties=properties,
            save_format="nii.gz",
        )
        return
    save_prediction_from_logits(logits, str(output_path), properties=properties)


class WritePredictionFromLogits(BasePredictionWriter):
    def __init__(self, output_dir, write_interval: str = "batch", save_format: str = ".nii.gz"):
        super().__init__(write_interval)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.save_format = save_format

    def write_on_batch_end(self, _trainer, _pl_module, data_dict, _batch_indices, _batch, _batch_idx, _dataloader_idx):
        # this will create N (num processes) files in `output_dir` each containing
        # the predictions of it's respective rank
        logits, properties, case_id = (
            data_dict["logits"],
            data_dict["properties"],
            data_dict["id"],
        )

        _write_prediction(
            logits,
            os.path.join(self.output_dir, case_id + self.save_format),
            properties=properties,
        )
        del data_dict
