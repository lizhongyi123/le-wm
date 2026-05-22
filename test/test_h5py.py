import h5py
import hdf5plugin
h5_path = r"F:\save\word_model\lewd2\train_data\pusht_expert_train.h5"

with h5py.File(h5_path, "r") as f:
    print("keys:", list(f.keys()))

    d = f["pixels"]
    print("pixels shape:", d.shape)
    print("pixels dtype:", d.dtype)
    print("pixels chunks:", d.chunks)
    print("pixels compression:", d.compression)
    print("pixels external:", d.external)

    x = d[0:20]
    print("read pixels[0:20] ok:", x.shape, x.dtype)

    x = d[1868112:1868132]
    print("read failed-index slice ok:", x.shape, x.dtype)