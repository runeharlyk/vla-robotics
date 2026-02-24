import h5py

def walk(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"{name}: shape={obj.shape}, dtype={obj.dtype}")
    else:
        print(f"{name}/")

with h5py.File("smolvla_language_pilot/trajectory.h5", "r") as f:
    f.visititems(walk)