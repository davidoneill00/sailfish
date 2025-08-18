import pickle
from pickle import load, dump
from sys import path, argv
from pathlib import Path

path.append(str(Path(__file__).parent.parent))

class FixNumpyCoreUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)
    
def load_checkpoint(filename, require_solver=None):
    with open(filename, "rb") as f:
        chkpt = FixNumpyCoreUnpickler(f).load()
    return chkpt

def scale_by_n(infile, n):
    #with open(infile, "rb") as inf:
    #    c = load(inf)
    c = load_checkpoint(infile)

    c["solution"] = c["solution"].repeat(n, axis=0).repeat(n, axis=1)
    c["driver"] = c["driver"]._replace(resolution=c["driver"].resolution * n)
    c["mesh"] = c["mesh"]._replace(ni=c["mesh"].ni * n, nj=c["mesh"].nj * n)
    outfile = infile.replace(".pk", "_upsampled.pk")

    with open(outfile, "wb") as f:
        dump(c, f)

    print(f"write {infile} scaled by {n}x to {outfile}")


if __name__ == "__main__":
    if len(argv) < 2:
        print("No files provided, use 'python3 upsample.py file1 file2 ...'")
    for file in argv[1:]:
        scale_by_n(file, 2)
