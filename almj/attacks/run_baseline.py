import asyncio
import dataclasses
import logging
import time
from pathlib import Path

import pandas as pd
import simple_parsing
from tqdm import tqdm

from almj.data_models.utils import RecitationRateFailureError
from almj.utils.experiment_utils import ExperimentConfigBase
from almj.utils.jailbreak_metrics import JailbreakMetrics

LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass
class ExperimentConfig(ExperimentConfigBase):
    input_file: Path | str
    model_id: str
    temperature: float = 1
    n_samples: int = 120
    audio_input: str = "audio_file"
    text_input: str = "none"
    run_by_prompt: bool = True
    experiment_filter: list[str] = None
    start_step: int = 0
    stop_idx: int = None

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.input_file, Path):
            self.input_file = Path(self.input_file)
        if "flash" in self.model_id:
            self.model_name = "flash"
        elif "pro" in self.model_id:
            self.model_name = "pro"

        if self.text_input == "none":
            self.text_input = None
        if self.audio_input == "none":
            self.audio_input = None


async def main(cfg: ExperimentConfig):
    jailbreak_metrics = JailbreakMetrics(cfg.api, n_workers=2)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_json(cfg.input_file, lines=True)
    LOGGER.info(f"Number of prompts for high temperature sampling: {len(df)}")

    if cfg.stop_idx:
        df = df[: cfg.stop_idx]

    if cfg.experiment_filter:
        LOGGER.info(f"Filtering for experiment: {cfg.experiment_filter}")
        assert all(
            experiment in df["experiment"].unique() for experiment in cfg.experiment_filter
        ), f"Experiment filter {cfg.experiment_filter} not in dataframe"
        df = df[df["experiment"].isin(cfg.experiment_filter)]

    if cfg.run_by_prompt:
        n_samples = 1
        iters = cfg.n_samples
        start = cfg.start_step
    else:
        n_samples = cfg.n_samples
        iters = 1
        start = cfg.start_step

    # Run through this in batches so we we're intermittently saving out outputs
    for i in tqdm(range(start, iters)):
        if iters > 1:
            output_file = cfg.model_name + "_" + cfg.input_file.stem + f"_high_temp{cfg.temperature}_step{i}.jsonl"
        else:
            output_file = cfg.model_name + "_" + cfg.input_file.stem + f"_high_temp{cfg.temperature}.jsonl"

        LOGGER.info(f"Getting results for step {i} on input file {cfg.input_file}")
        LOGGER.info(f"Outputting to {cfg.output_dir/output_file}")
        while True:
            try:
                df_out = await jailbreak_metrics.high_temperature_sampling(
                    dataset=df,
                    model=cfg.model_id,
                    n_samples=n_samples,
                    temperature=cfg.temperature,
                    input_key=cfg.text_input,
                    audio_key=cfg.audio_input,
                    output_key="proportion_flagged_audio",
                )
            except RecitationRateFailureError as e:
                print(f"{e}. Sleeping for 2 minutes.")
                time.sleep(30)
            else:
                break

        print(f"Saving {cfg.output_dir/output_file}")
        df_out.to_json(cfg.output_dir / output_file, orient="records", lines=True)


if __name__ == "__main__":
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(ExperimentConfig, dest="experiment_config")
    args = parser.parse_args()
    cfg: ExperimentConfig = args.experiment_config
    cfg.setup_experiment(log_file_prefix="run-repeated-sampling")
    asyncio.run(main(cfg))
