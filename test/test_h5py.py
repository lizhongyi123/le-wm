import h5py
import hdf5plugin
h5_path = r"C:\Users\wangh\Desktop\world_model\lewd2\train_data\pusht_expert_train.h5"

with h5py.File(h5_path, "r") as f:
    print("keys:", list(f.keys()))

    d = f["pixels"]
    print("pixels shape:", d.shape)
    # print(d[1])

    print("  " * 100)

    d = f["action"]
    print("action shape:", d.shape)
    print(d[5: 10])
    
    print("  " * 100)

    d = f["ep_len"]
    print("ep_len shape:", d.shape)
    print(d[: 7])

    print("  " * 100)

    d = f["ep_offset"]
    print("ep_offset shape:", d.shape)
    print(d[:7])

