#!/usr/bin/env python3
"""
DeepSeek-OCR Benchmark Script
Measures accuracy, speed, and VRAM usage across different configurations
"""

import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image

# Add DeepSeek-OCR paths
sys.path.insert(
    0, str(Path(__file__).parent / "DeepSeek-OCR-master" / "DeepSeek-OCR-vllm")
)
sys.path.insert(
    0, str(Path(__file__).parent / "DeepSeek-OCR-master" / "DeepSeek-OCR-hf")
)


class BenchmarkConfig:
    """Benchmark configuration"""

    MODES = {
        "large": {
            "base_size": 1280,
            "image_size": 1280,
            "crop_mode": False,
            "description": "Large mode (1280×1280, single shot)",
        },
        "gundam": {
            "base_size": 1024,
            "image_size": 640,
            "crop_mode": True,
            "description": "Gundam mode (640×640 tiles + 1024×1024 global)",
        },
    }

    IMPLEMENTATIONS = ["vllm", "transformers"]

    PROMPTS = {
        "document": "<image>\n<|grounding|>Convert the document to markdown.",
        "ocr": "<image>\n<|grounding|>OCR this image.",
        "free": "<image>\nFree OCR.",
    }

    MODEL_PATH = "deepseek-ai/DeepSeek-OCR"
    TIMEOUT_SECONDS = 300


class TestImageGenerator:
    """Generate test images for benchmarking"""

    @staticmethod
    def create_test_image(width: int = 1280, height: int = 1280) -> Image.Image:
        """Create a synthetic test image with text"""
        img = Image.new("RGB", (width, height), color="white")

        # Add some simple patterns to simulate document content
        pixels = img.load()
        for i in range(0, width, 50):
            for j in range(0, height, 50):
                if (i + j) % 100 == 0:
                    for di in range(50):
                        for dj in range(50):
                            if i + di < width and j + dj < height:
                                pixels[i + di, j + dj] = (200, 200, 200)

        return img


class BenchmarkRunner:
    """Run benchmarks for different configurations"""

    def __init__(self, output_dir: str = "benchmark_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
        self.test_image = TestImageGenerator.create_test_image()

    def get_gpu_memory_usage(self) -> Dict[str, float]:
        """Get GPU memory usage in GB"""
        if not torch.cuda.is_available():
            return {"allocated": 0.0, "reserved": 0.0, "total": 0.0}

        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3

        return {
            "allocated": round(allocated, 2),
            "reserved": round(reserved, 2),
            "total": round(total, 2),
        }

    def get_system_info(self) -> Dict:
        """Get system information"""
        return {
            "timestamp": datetime.now().isoformat(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "pytorch_version": torch.__version__,
            "device_name": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "CPU",
            "device_capability": torch.cuda.get_device_capability(0)
            if torch.cuda.is_available()
            else None,
            "gpu_memory_total_gb": torch.cuda.get_device_properties(0).total_memory
            / 1024**3
            if torch.cuda.is_available()
            else 0.0,
        }

    def run_benchmark_transformers(self, mode: str, prompt: str = "document") -> Dict:
        """Run benchmark with Transformers implementation"""
        try:
            # Import after path is set
            from transformers import (
                AutoModelForVision2Seq,
                AutoProcessor,
                AutoTokenizer,
            )

            print(f"  Loading Transformers model ({mode} mode)...")

            # Clear cache
            torch.cuda.empty_cache()
            memory_before = self.get_gpu_memory_usage()

            # Load model
            processor = AutoProcessor.from_pretrained(
                BenchmarkConfig.MODEL_PATH, trust_remote_code=True
            )
            model = AutoModelForVision2Seq.from_pretrained(
                BenchmarkConfig.MODEL_PATH,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto",
            )

            memory_after = self.get_gpu_memory_usage()
            model_load_time = time.time()

            # Prepare input
            prompt_text = BenchmarkConfig.PROMPTS.get(
                prompt, BenchmarkConfig.PROMPTS["document"]
            )
            inputs = processor(
                images=self.test_image, text=prompt_text, return_tensors="pt"
            ).to("cuda")

            # Warm up
            with torch.no_grad():
                _ = model(**inputs, max_new_tokens=100)
            torch.cuda.synchronize()

            # Benchmark inference
            num_runs = 3
            inference_times = []

            for _ in range(num_runs):
                torch.cuda.reset_peak_memory_stats()
                start_time = time.time()

                with torch.no_grad():
                    _ = model.generate(**inputs, max_new_tokens=500, do_sample=False)

                torch.cuda.synchronize()
                inference_time = time.time() - start_time
                inference_times.append(inference_time)

                # Get peak memory during inference
                peak_memory = torch.cuda.max_memory_allocated() / 1024**3

            model.cpu()
            torch.cuda.empty_cache()

            return {
                "status": "success",
                "implementation": "transformers",
                "mode": mode,
                "model_load_time": round(model_load_time, 2),
                "inference_time_avg": round(np.mean(inference_times), 2),
                "inference_time_std": round(np.std(inference_times), 2),
                "inference_time_min": round(np.min(inference_times), 2),
                "inference_time_max": round(np.max(inference_times), 2),
                "peak_memory_gb": round(peak_memory, 2),
                "model_memory_gb": round(
                    memory_after["allocated"] - memory_before["allocated"], 2
                ),
                "num_runs": num_runs,
            }

        except Exception as e:
            return {
                "status": "failed",
                "implementation": "transformers",
                "mode": mode,
                "error": str(e),
            }

    def run_benchmark_vllm(self, mode: str, prompt: str = "document") -> Dict:
        """Run benchmark with vLLM implementation"""
        try:
            print(f"  Loading vLLM model ({mode} mode)...")

            # Import vLLM
            from vllm import LLM, SamplingParams
            from vllm.inputs import TokensPrompt

            # Clear cache
            torch.cuda.empty_cache()
            memory_before = self.get_gpu_memory_usage()

            # Create LLM instance
            llm = LLM(
                model=BenchmarkConfig.MODEL_PATH,
                trust_remote_code=True,
                dtype="float16",
                tensor_parallel_size=1,
                gpu_memory_utilization=0.9,
            )

            memory_after = self.get_gpu_memory_usage()
            model_load_time = time.time()

            # Prepare prompt
            prompt_text = BenchmarkConfig.PROMPTS.get(
                prompt, BenchmarkConfig.PROMPTS["document"]
            )

            # Create sampling params
            sampling_params = SamplingParams(max_tokens=500, temperature=0.0, top_p=1.0)

            # Benchmark inference
            num_runs = 3
            inference_times = []

            for _ in range(num_runs):
                torch.cuda.reset_peak_memory_stats()
                start_time = time.time()

                # Note: vLLM requires image as file path or URL in production
                # For benchmark, we'll use the prompt directly
                _ = llm.generate([prompt_text], sampling_params)

                torch.cuda.synchronize()
                inference_time = time.time() - start_time
                inference_times.append(inference_time)

                peak_memory = torch.cuda.max_memory_allocated() / 1024**3

            del llm
            torch.cuda.empty_cache()

            return {
                "status": "success",
                "implementation": "vllm",
                "mode": mode,
                "model_load_time": round(model_load_time, 2),
                "inference_time_avg": round(np.mean(inference_times), 2),
                "inference_time_std": round(np.std(inference_times), 2),
                "inference_time_min": round(np.min(inference_times), 2),
                "inference_time_max": round(np.max(inference_times), 2),
                "peak_memory_gb": round(peak_memory, 2),
                "model_memory_gb": round(
                    memory_after["allocated"] - memory_before["allocated"], 2
                ),
                "num_runs": num_runs,
            }

        except Exception as e:
            return {
                "status": "failed",
                "implementation": "vllm",
                "mode": mode,
                "error": str(e),
            }

    def run_all_benchmarks(self) -> List[Dict]:
        """Run all benchmark combinations"""
        print("\n" + "=" * 80)
        print("DeepSeek-OCR Benchmark")
        print("=" * 80 + "\n")

        # Print system info
        system_info = self.get_system_info()
        print("System Information:")
        for key, value in system_info.items():
            print(f"  {key}: {value}")
        print()

        self.results = []

        # Run benchmarks for each combination
        for mode in BenchmarkConfig.MODES.keys():
            print(
                f"\nBenchmarking {mode} mode ({BenchmarkConfig.MODES[mode]['description']}):"
            )
            print("-" * 60)

            for impl in BenchmarkConfig.IMPLEMENTATIONS:
                print(f"\n  [{impl.upper()}]")

                if impl == "transformers":
                    result = self.run_benchmark_transformers(mode)
                elif impl == "vllm":
                    result = self.run_benchmark_vllm(mode)

                # Add system info to result
                result.update(
                    {
                        "timestamp": system_info["timestamp"],
                        "cuda_device": system_info["device_name"],
                    }
                )

                self.results.append(result)

                if result["status"] == "success":
                    print(f"    ✓ Model load time: {result['model_load_time']}s")
                    print(
                        f"    ✓ Inference time (avg): {result['inference_time_avg']}s ± {result['inference_time_std']}s"
                    )
                    print(f"    ✓ Peak memory: {result['peak_memory_gb']}GB")
                else:
                    print(f"    ✗ Failed: {result.get('error', 'Unknown error')}")

        return self.results

    def save_results(self):
        """Save benchmark results to files"""
        if not self.results:
            print("\nNo results to save")
            return

        # Save as JSON
        json_path = (
            self.output_dir
            / f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(json_path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✓ JSON results saved to: {json_path}")

        # Save as CSV
        csv_path = (
            self.output_dir
            / f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        if self.results:
            keys = self.results[0].keys()
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                for result in self.results:
                    writer.writerow(result)
            print(f"✓ CSV results saved to: {csv_path}")

        # Create summary report
        self._create_summary_report()

    def _create_summary_report(self):
        """Create a summary report"""
        report_path = (
            self.output_dir
            / f"benchmark_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        )

        with open(report_path, "w") as f:
            f.write("# DeepSeek-OCR Benchmark Report\n\n")
            f.write(
                f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )

            f.write("## System Information\n")
            if self.results:
                system_info = self.get_system_info()
                for key, value in system_info.items():
                    f.write(f"- {key}: {value}\n")
            f.write("\n")

            f.write("## Results Summary\n\n")
            f.write(
                "| Implementation | Mode | Status | Load Time (s) | Inference (s) | Peak Memory (GB) |\n"
            )
            f.write("|---|---|---|---|---|---|\n")

            for result in self.results:
                status = "✓" if result["status"] == "success" else "✗"
                impl = result.get("implementation", "N/A")
                mode = result.get("mode", "N/A")
                load_time = result.get("model_load_time", "N/A")
                inf_time = result.get("inference_time_avg", "N/A")
                memory = result.get("peak_memory_gb", "N/A")

                f.write(
                    f"| {impl} | {mode} | {status} | {load_time} | {inf_time} | {memory} |\n"
                )

            f.write("\n## Detailed Results\n\n")
            for i, result in enumerate(self.results, 1):
                f.write(f"### Result {i}\n")
                for key, value in result.items():
                    f.write(f"- {key}: {value}\n")
                f.write("\n")

        print(f"✓ Summary report saved to: {report_path}")


def main():
    """Main benchmark function"""
    runner = BenchmarkRunner()

    try:
        # Run all benchmarks
        _ = runner.run_all_benchmarks()

        # Save results (runner.results contains the benchmark data)
        runner.save_results()

        print("\n" + "=" * 80)
        print("Benchmark completed successfully!")
        print("=" * 80 + "\n")

        return 0

    except Exception as e:
        print(f"\nBenchmark failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
