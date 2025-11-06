import numpy as np
from pathlib import Path
import argparse
import pickle as pk
import sys
sys.path.insert(1,"/groups/astro/davidon/sailfish/")
import sailfish
import subprocess
sys.path.insert(1, ".")


class FixNumpyCoreUnpickler(pk.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)

def load_checkpoint(filename, require_solver=None):
    with open(filename, "rb") as f:
        chkpt = FixNumpyCoreUnpickler(f).load()
    return chkpt


def file_load(indir, movie_outdir, savefigbool, filename):
    file_count        = 0
    current_path_name = Path().resolve()

    if current_path_name.name != "sailfish":
        raise RuntimeError(f"Script was designed to be run from 'sailfish', but you're in: {current_path_name}")

    frame_list        = []
    Path('{}/output-figures'.format(current_path_name)).mkdir(parents=True, exist_ok=True)
    
    for name in sorted(Path(indir).iterdir()):
        file_count += 1

        chkpt       = load_checkpoint(name)
        CurrentTime = chkpt["time"]/ 2 / np.pi
        
        plot_script = str(current_path_name / "plot.py")
        plot_args = [
            "python", plot_script,
            name,
            "-f", str('t4'),
            "-l",           
            "--radius", str(4.0),
            #"--vmap",
            "--vmin", str(23.5),
            "--vmax", str(25),
            "-o", "output-figures"
        ]

        subprocess.run(plot_args, check=True)
        
        SavedFileName  = f"DensityMap-{int(CurrentTime * 100):05d}.png"
        frame_list.append(SavedFileName)
        

    with open("output-figures/frames.txt", "w") as f:
        for fname in frame_list:
            f.write(f"file '{fname}'\n")


    from shutil import move

    # After all images are saved, rename them sequentially
    for i, fname in enumerate(sorted(Path("output-figures").glob("DensityMap-*.png"))):
        new_name = Path("output-figures") / f"DensityMap-{i:05d}.png"
        move(fname, new_name)
    

    make_movie(current_path_name, movie_outdir, filename)
    #if savefigbool is False:
    #    os.system("rm -rf {}/{}".format(current_path_name, 'output-figures'))




def make_movie(current_path, movie_outdir, filename, frame_list=None, use_glob=False):
    output_dir = Path(current_path) / movie_outdir
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{filename}.mp4"

    # Choose input: contiguous sequence (%05d) or glob for gaps
    if use_glob:
        # Works even if some frames are missing (zero-padded names keep correct order)
        input_args = ["-pattern_type", "glob",
                      "-i", str(Path(current_path) / "output-figures" / "DensityMap-*.png")]
    elif frame_list:
        # Explicit file list for arbitrary ordering / mixed padding
        list_file = output_dir / "frames.txt"
        with list_file.open("w") as f:
            for p in frame_list:
                f.write(f"file '{p}'\n")
                f.write("duration 0.1\n")  # 10 fps; adjust as needed
            if frame_list:
                f.write(f"file '{frame_list[-1]}'\n")  # repeat last file per concat spec
        input_args = ["-f", "concat", "-safe", "0", "-i", str(list_file)]
    else:
        # Contiguous %05d sequence starting at 0 (change start_number if needed)
        input_args = ["-start_number", "0",
                      "-i", str(Path(current_path) / "output-figures" / "DensityMap-%05d.png")]

    cmd = [
        "ffmpeg", "-y",
        "-framerate", "30",
        *input_args,
        # Single filter chain: make dimensions even + slow to 0.5x
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,setpts=2*PTS",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_file),
    ]
    subprocess.run(cmd, check=True)






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', default='', help='Checkpoint file directory.', required=True)
    parser.add_argument('--outdir', default='movie', help='Output movie directory.')
    parser.add_argument('--filename', default='movie', help='Output movie name.')
    parser.add_argument('--savefigs', default=False, help='Whether the program saves the figures used to make the movie.')
    args = parser.parse_args()

    file_load(args.indir, args.outdir, args.savefigs, args.filename)
