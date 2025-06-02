import os
import logging
import subprocess
import time
from pathlib import Path
import threading
import shutil
import pandas as pd
from utils import setup_logging, PROJECT_PATH, model_name_url_list
import json

setup_logging()


def custom_sort_key(folder_name):
    """Sort by the number in folder name"""
    parts = folder_name.split("__")
    try:
        return int(parts[0])
    except Exception:
        return 0


defect_csv = f"{PROJECT_PATH}/datasets/llmdd_defect.csv"
defect_info = pd.read_csv(defect_csv)
defect_info.set_index("id", inplace=True)


class MainSingleProcessor(threading.Thread):
    def __init__(self, model_name, model_url, input_dir, output_base_dir, folder_names):
        super().__init__(name=f"Thread-{model_name}")
        self.model_name = model_name
        self.model_url = model_url
        self.input_dir = Path(input_dir)
        self.output_base_dir = Path(output_base_dir)
        self.folder_names = folder_names

    def process_project(self, folder_name):
        repo_dir = self.input_dir / folder_name
        output_dir = self.output_base_dir / self.model_name / folder_name

        # read defect_info
        defect_id = int(folder_name.split("__")[0])
        if defect_id in defect_info.index:
            row = defect_info.loc[defect_id]
            explanation = str(row["explanation"])
            defect_tests = str(row["defect-triggering tests"])
            consequences = str(row["consequences_details"])
        else:
            explanation = defect_tests = consequences = ""

        print(explanation, defect_tests, consequences)

        # continue if results already exist
        if os.path.exists(output_dir):
            defect_results_path = output_dir / "defect_results.json"
            calls_llm_path = output_dir / "calls_llm.jsonl"
            log_file_path = output_dir / "running.log"

            # first check if there is any error in the log file
            # with open(log_file_path, "r", encoding="utf-8") as f:
            #     for line in f:
            #         #  or "[WARNING]" in line
            #         if "[ERROR]" in line:
            #             shutil.rmtree(output_dir)
            #             break
            if os.path.exists(output_dir):
                # if there is no error, check if the results already exist
                if defect_results_path.exists() and calls_llm_path.exists():
                    return
                #     with open(defect_results_path, "r", encoding="utf-8") as f:
                #         defect_data = json.load(f)
                #     # check if both JSONs are not empty
                #     if defect_data:
                #         logging.info(f"[{self.model_name}] Results already exist for {folder_name}, skipping...")
                #         return
                else:
                    shutil.rmtree(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"[{self.model_name}] Processing project: {folder_name}")

        try:
            cmd = [
                "python3",
                "run_main_single.py",
                "--repo_dir",
                str(repo_dir),
                "--defect_explanation",
                explanation,
                "--defect_consequences",
                consequences,
                "--defect_tests",
                defect_tests,
                "--output_dir",
                str(output_dir),
                "--base_url",
                self.model_url,
                "--api_key",
                "token-123",
                "--model",
                self.model_name,
                "--embedding_url",
                "http://localhost:11434/v1",
                "--embedding_key",
                "ollama",
                "--embedding_model",
                "bge-m3:567m-fp16",
                "--static_threshold",
                "0.01",
                "--embedding_threshold",
                "0.3",
                "--max_candidates_per_file",
                "1000",
                "--max_candidates_for_context",
                "1000",
                "--context_hops",
                "1",
                "--top_files_count",
                "3",
                "--force_rebuild_kg",
                "0",
            ]

            command_file_path = output_dir / "command_config.txt"
            with open(command_file_path, "w") as f:
                f.write(" ".join(cmd))

            start_time = time.time()
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            duration = time.time() - start_time

            logging.info(f"[{self.model_name}] Successfully processed {folder_name} in {duration:.1f}s")

        except Exception as e:
            logging.error(f"[{self.model_name}] Error processing {folder_name}: {str(e)}")

        time.sleep(5)

    def run(self):
        for folder_name in self.folder_names:
            self.process_project(folder_name)
        logging.info(f"[{self.model_name}] Completed processing all projects")


def run_main_single(input_dir, output_base_dir, model_name_url_list):
    folder_names = os.listdir(input_dir)
    folder_names = [f for f in folder_names if (Path(input_dir) / f).is_dir()]
    folder_names.sort(key=custom_sort_key)
    # folder_names = folder_names[:2]

    threads = []
    for model_name, model_url in model_name_url_list:
        processor = MainSingleProcessor(model_name, model_url, input_dir, output_base_dir, folder_names.copy())
        threads.append(processor)
        processor.start()

    for thread in threads:
        thread.join()
    logging.info("All models have completed processing")


def main():
    input_dir = f"{PROJECT_PATH}/datasets/llmdd"
    output_base_dir = f"{PROJECT_PATH}/results/main_output"
    os.makedirs(output_base_dir, exist_ok=True)

    model_name_url_list = model_name_url_list

    run_main_single(input_dir, output_base_dir, model_name_url_list)


if __name__ == "__main__":
    main()
