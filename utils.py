from tensorflow.keras.layers import DepthwiseConv2D
import h5py, json, tempfile, shutil

# Custom DepthwiseConv2D to ignore unsupported "groups" argument
class CompatibleDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, groups=None, **kwargs):
        super().__init__(*args, **kwargs)


# Remove "groups" key from corrupted legacy Keras model configs
def _remove_groups(obj):
    if isinstance(obj, dict):
        obj.pop("groups", None)
        for k, v in obj.items():
            obj[k] = _remove_groups(v)
    elif isinstance(obj, list):
        return [_remove_groups(i) for i in obj]
    return obj


# Clean the H5 model file and return a temporary cleaned version
def strip_groups_from_h5(original_path):
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".h5")
    temp.close()
    shutil.copy2(original_path, temp.name)

    with h5py.File(temp.name, "r+") as f:
        # model_config may be in attrs OR dataset
        if "model_config" in f.attrs:
            raw = f.attrs["model_config"]
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            cfg = json.loads(raw)
            cleaned = _remove_groups(cfg)
            f.attrs["model_config"] = json.dumps(cleaned)

        elif "model_config" in f:
            raw = f["model_config"][()]
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            cfg = json.loads(raw)
            cleaned = _remove_groups(cfg)
            del f["model_config"]
            f.create_dataset("model_config", data=json.dumps(cleaned))

    return temp.name
