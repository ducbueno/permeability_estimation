import shutil
import subprocess
from pathlib import Path

import numpy as np
import polars as pl


class LBPMManager:
    def __init__(self, **kwargs):
        script_dir = Path(__file__).parent.absolute()
        self.work_dir = script_dir / "work"

        self.sample_size = kwargs.get("sample_size", [])
        self.mirror = kwargs.get("mirror", True)
        self.vl = kwargs.get("vl", 1.0)
        self.tau = kwargs.get("tau", 1.0)
        self.force = kwargs.get("force", [0, 0, 1e-8])
        self.tsmax = kwargs.get("tsmax", 4000)
        self.tol = kwargs.get("tol", 0.01)
        self.protocol = kwargs.get("protocol", "mrt")

        self.input_path = self.work_dir / "input.db"
        self.awk_script_path = script_dir / "gen_input_db.awk"
        self.template_path = script_dir / "templates" / f"input_{self.protocol}.db"

    def _assemble_input(self, domain):
        sample_size_str = ", ".join(map(str, reversed(domain.shape)))
        force_str = ", ".join(map(str, self.force))

        with (
            open(self.template_path) as template_file,
            open(self.input_path, "w") as output_file,
        ):
            subprocess.run(
                [
                    str(self.awk_script_path),
                    "-v",
                    f"tau={self.tau}",
                    "-v",
                    f"vl={self.vl}",
                    "-v",
                    f"tsmax={self.tsmax}",
                    "-v",
                    f"tol={self.tol}",
                    "-v",
                    f"force={force_str}",
                    "-v",
                    f"sample_size={sample_size_str}",
                    "-v",
                    f"subdomain_size={sample_size_str}",
                    "-v",
                    f"mirror={self.mirror}",
                ],
                stdin=template_file,
                stdout=output_file,
                check=True,
                text=True,
            )

    def run_simulation(self, domain):
        self.work_dir.mkdir(exist_ok=True)

        if self.mirror:
            domain = np.append(domain, np.flip(domain[:-1, :, :], axis=0), axis=0)

        with open(self.work_dir / "domain.raw", "wb") as f:
            domain.tofile(f)

        self._assemble_input(domain)

        cmd = str(
            Path.home()
            / "src/lbpm/lbpm/build/bin"
            / (
                "lbpm_permeability_simulator"
                if self.protocol == "mrt"
                else "lbpm_greyscale_simulator"
            )
        )

        try:
            result = subprocess.run(
                [cmd, str(self.work_dir / "input.db")],
                cwd=self.work_dir,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            result = e.stdout

        lbpm_porosity = next(
            (
                float(line.split(" = ")[1].strip())
                for line in result.stdout.split("\n")
                if "Media porosity" in line
            ),
            None,
        )
        permeability = (
            pl.scan_csv(self.work_dir / "Permeability.csv", separator=" ")
            .select("absperm")
            .tail(1)
            .collect()
            .item()
        )

        shutil.rmtree(self.work_dir)

        return {"lbpm_porosity": lbpm_porosity, "permeability": permeability}
