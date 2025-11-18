from pathlib import Path
import argparse
import pickle as pk
import sys
import numpy as np

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
import subprocess
from shutil import move, rmtree


def load_checkpoint(filename, require_solver=None):
    with open(filename, "rb") as f:
        chkpt = pk.load(file=f)
    return chkpt


def file_load(indir, movie_outdir, savefigbool, filename, quick_plotting):
    current_path_name = Path().resolve()
    Path('{}/output-figures/'.format(current_path_name)).mkdir(parents=True, exist_ok=True)
    
    for name in sorted(Path(indir).iterdir()):
        if not quick_plotting:
            chkpt       = load_checkpoint(name)
            CurrentTime = chkpt["time"]/ 2 / np.pi
            
            plot_script = str(current_path_name / "plot.py")
            plot_args = [
                "python", plot_script,
                name,
                #"-f", str('t4'),
                "-l",           
                "--radius", str(4.0),
                "--vmap",
                "--vmin", str(-8),
                #"--vmax", str(25),
                "-o", "output-figures/"
            ]

            subprocess.run(plot_args, check=True)

    # After all images are saved, rename them sequentially
    ordered_frames = sorted(Path("output-figures").glob("DensityMap-*.png"))
    if not ordered_frames:
        raise FileNotFoundError("No frames were generated. Did you run with --quick_plotting?")

    for i, fname in enumerate(ordered_frames):
        new_name = Path("output-figures") / f"DensityMap-{i:05d}.png"
        if fname != new_name:
            move(fname, new_name)
    

    make_movie(current_path_name, movie_outdir, filename)
    if savefigbool is False:
        rmtree(Path(current_path_name) / "output-figures", ignore_errors=True)




def make_movie(current_path, movie_outdir, filename):
    output_dir  = Path(current_path) / movie_outdir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{filename}.mp4"

    input_pattern = Path(current_path) / "output-figures" / "DensityMap-%05d.png"

    cmd = [
        "ffmpeg", "-y",
        "-framerate", "30",
        "-start_number", "0",
        "-i", str(input_pattern),
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
    parser.add_argument("indir", help='Checkpoint file directory.')
    parser.add_argument('--outdir', '-o', default='movie', help='Output movie directory.')
    parser.add_argument('--filename', default='movie', help='Output movie name.')
    parser.add_argument('--savefigs', default=True, help='Whether the program saves the figures used to make the movie.')
    parser.add_argument('--quick_plotting', '-q', action='store_true', help='Whether to run the plotting script.')
    args = parser.parse_args()

    print(args.indir)

    file_load(args.indir, args.outdir, args.savefigs, args.filename, args.quick_plotting)
