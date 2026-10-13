"""Check out only K1 assets and upstream license/docs at the validated revision."""
from pathlib import Path
import subprocess

REVISION = "12a97516f57469078707afb587a5f84c1cb110d1"
DEST = Path(__file__).resolve().parents[1] / "assets/booster_assets"

if __name__ == "__main__":
    if DEST.exists():
        current = subprocess.check_output(["git", "-C", str(DEST), "rev-parse", "HEAD"], text=True).strip()
        if current != REVISION:
            raise SystemExit(f"Existing checkout is {current}; expected {REVISION}. Existing files left intact.")
    else:
        DEST.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout",
                        "https://github.com/BoosterRobotics/booster_assets.git", str(DEST)], check=True)
    subprocess.run(["git", "-C", str(DEST), "sparse-checkout", "set", "--no-cone",
                    "/robots/K1/", "/LICENSE", "/README.md"], check=True)
    if subprocess.check_output(["git", "-C", str(DEST), "rev-parse", "HEAD"], text=True).strip() != REVISION:
        subprocess.run(["git", "-C", str(DEST), "checkout", "--detach", REVISION], check=True)
    # Cache one upstream metadata blob for the offline audit; no T1/T2 files are checked out.
    subprocess.run(["git", "-C", str(DEST), "show", f"{REVISION}:src/booster_assets/motions.py"],
                   stdout=subprocess.DEVNULL, check=True)
    print(f"Official assets: {DEST} @ {REVISION}")
