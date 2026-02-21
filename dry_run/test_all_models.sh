#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Test script for all supported models in vLLM-Omni.
#
# Each model has its own test function that calls the corresponding example script
# from vllm-omni/examples/offline_inference/.
#
# Features:
# - Automatic model path resolution: checks local paths first, downloads if needed
# - Custom model path mapping: edit MODEL_PATH_MAP to use local model weights
# - Automatic download: uses huggingface-cli to download missing models
#
# Usage:
#     ./test_all_models.sh --model qwen3_omni_30b
#     ./test_all_models.sh --model all  # Test all models
#     ./test_all_models.sh --model qwen_image --model-cache-dir /path/to/models

set -euo pipefail

# Get script directory and vllm-omni root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_OMNI_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EXAMPLES_DIR="$VLLM_OMNI_ROOT/examples/offline_inference"

# Default model cache directory
DEFAULT_MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-/storage/omni_models}"

# Default values
MODEL="all"
OUTPUT_DIR="outputs"
MODEL_CACHE_DIR="$DEFAULT_MODEL_CACHE_DIR"
ENABLE_PROFILE=false
ENFORCE_EAGER=false
USE_HF_MODEL_PATH=false
DOWNLOAD_ONLY=false
FORCE_DOWNLOAD=false
ENABLE_DEBUG=false

# Model name to HuggingFace model identifier mapping
declare -A MODEL_NAME_MAP=(
    ["qwen3_omni_30b"]="Qwen/Qwen3-Omni-30B-A3B-Instruct"
    ["qwen2_5_omni_7b"]="Qwen/Qwen2.5-Omni-7B"
    ["qwen2_5_omni_3b"]="Qwen/Qwen2.5-Omni-3B"
    ["bagel_7b_mot_text2img"]="ByteDance-Seed/BAGEL-7B-MoT"
    ["bagel_7b_mot_img2img"]="ByteDance-Seed/BAGEL-7B-MoT"
    ["qwen_image"]="Qwen/Qwen-Image"
    ["qwen_image_2512"]="Qwen/Qwen-Image-2512"
    ["qwen_image_edit"]="Qwen/Qwen-Image-Edit"
    ["qwen_image_edit_2509"]="Qwen/Qwen-Image-Edit-2509"
    ["qwen_image_layered"]="Qwen/Qwen-Image-Layered"
    ["z_image_turbo"]="Tongyi-MAI/Z-Image-Turbo"
    ["glm_image"]="zai-org/GLM-Image"
    ["ovis_image"]="AIDC-AI/Ovis-Image-7B"
    ["longcat_image"]="meituan-longcat/LongCat-Image"
    ["longcat_image_edit"]="meituan-longcat/LongCat-Image-Edit"
    ["hunyuan_image_3_0"]="tencent/HunyuanImage-3.0"
    ["stable_diffusion_3_5_medium"]="stabilityai/stable-diffusion-3.5-medium"
    ["flux_2_klein_4b"]="black-forest-labs/FLUX.2-klein-4B"
    ["flux_2_klein_9b"]="black-forest-labs/FLUX.2-klein-9B"
    ["flux_1_dev"]="black-forest-labs/FLUX.1-dev"
    ["wan2_2_t2v_a14b"]="Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    ["wan2_2_ti2v_5b"]="Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    ["wan2_2_i2v_a14b"]="Wan-AI/Wan2.2-I2V-A14B-Diffusers"
    ["stable_audio_open"]="stabilityai/stable-audio-open-1.0"
    ["qwen3_tts_custom_voice"]="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    ["qwen3_tts_voice_design"]="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    ["qwen3_tts_base"]="Qwen/Qwen3-TTS-12Hz-0.6B-Base"
)

# Model path mapping (can be customized by users)
# Example: MODEL_PATH_MAP["Qwen/Qwen-Image"]="/path/to/local/qwen-image"
declare -A MODEL_PATH_MAP=(
    ["Qwen/Qwen3-Omni-30B-A3B-Instruct"]=""
    ["Qwen/Qwen2.5-Omni-7B"]=""
    ["Qwen/Qwen2.5-Omni-3B"]=""
    ["ByteDance-Seed/BAGEL-7B-MoT"]=""
    ["Qwen/Qwen-Image"]=""
    ["Qwen/Qwen-Image-2512"]=""
    ["Qwen/Qwen-Image-Edit"]=""
    ["Qwen/Qwen-Image-Edit-2509"]=""
    ["Qwen/Qwen-Image-Layered"]=""
    ["Tongyi-MAI/Z-Image-Turbo"]=""
    ["zai-org/GLM-Image"]=""
    ["AIDC-AI/Ovis-Image-7B"]=""
    ["meituan-longcat/LongCat-Image"]=""
    ["meituan-longcat/LongCat-Image-Edit"]=""
    ["tencent/HunyuanImage-3.0"]=""
    ["stabilityai/stable-diffusion-3.5-medium"]=""
    ["black-forest-labs/FLUX.2-klein-4B"]=""
    ["black-forest-labs/FLUX.2-klein-9B"]=""
    ["black-forest-labs/FLUX.1-dev"]=""
    ["Wan-AI/Wan2.2-T2V-A14B-Diffusers"]=""
    ["Wan-AI/Wan2.2-TI2V-5B-Diffusers"]=""
    ["Wan-AI/Wan2.2-I2V-A14B-Diffusers"]=""
    ["stabilityai/stable-audio-open-1.0"]=""
    ["Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"]=""
    ["Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"]=""
    ["Qwen/Qwen3-TTS-12Hz-0.6B-Base"]=""
)

# Test registry - list of enabled models (matches MODEL_TEST_REGISTRY in Python)
# Note: Some models may be commented out in the Python file
# ENABLED_MODELS=(
#     "qwen_image"
#     "qwen_image_edit"
#     "qwen_image_edit_2509"
#     "qwen_image_layered"
#     "z_image_turbo"
#     "glm_image"
#     "ovis_image"
#     "longcat_image"
#     "longcat_image_edit"
#     "hunyuan_image_3_0"
#     "stable_diffusion_3_5_medium"
#     "flux_2_klein_4b"
#     "flux_1_dev"
#     "wan2_2_t2v_a14b"
#     "wan2_2_ti2v_5b"
#     "wan2_2_i2v_a14b"
#     "stable_audio_open"
#     "qwen3_tts_custom_voice"
#     "qwen3_tts_voice_design"
#     "qwen3_tts_base"
#     "qwen3_omni_30b"
#     "qwen2_5_omni_7b"
#     "bagel_7b_mot_text2img"
#     "bagel_7b_mot_img2img"
# )
ENABLED_MODELS=(
    "wan2_2_t2v_a14b"
    "qwen2_5_omni_7b"
    "qwen3_omni_30b"
)

# Helper functions
print_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    --model MODEL              Model to test (use 'all' to test all models)
    --output-dir DIR           Base output directory (default: outputs)
    --model-cache-dir DIR      Custom directory for caching/downloading models
    --enable-profile           Enable PyTorch profiling
    --enforce-eager            Disable torch.compile and force eager execution
    --use-hf-model-path        Use HuggingFace model paths directly
    --enable-debug             Enable vLLM debug logging (sets VLLM_LOGGING_LEVEL=DEBUG)
    --download-only            Only download models without running tests
    --force-download           Force re-download even if models exist
    -h, --help                 Show this help message

Note: ZE_AFFINITY_MASK environment variable (if set) will be applied when running scripts.

Examples:
    $0 --model qwen_image
    $0 --model all --output-dir /tmp/test_outputs
    $0 --model qwen_image --use-hf-model-path
    $0 --download-only --model all
EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --model)
                MODEL="$2"
                shift 2
                ;;
            --output-dir)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            --model-cache-dir)
                MODEL_CACHE_DIR="$2"
                shift 2
                ;;
            --enable-profile)
                ENABLE_PROFILE=true
                shift
                ;;
            --enforce-eager)
                ENFORCE_EAGER=true
                shift
                ;;
            --use-hf-model-path)
                USE_HF_MODEL_PATH=true
                shift
                ;;
            --enable-debug)
                ENABLE_DEBUG=true
                shift
                ;;
            --download-only)
                DOWNLOAD_ONLY=true
                shift
                ;;
            --force-download)
                FORCE_DOWNLOAD=true
                shift
                ;;
            -h|--help)
                print_usage
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                print_usage
                exit 1
                ;;
        esac
    done
}

# Get model path, downloading if necessary
# Outputs only the model path to stdout, all other messages to stderr
get_model_path() {
    local model_name="$1"
    local cache_dir="${2:-$MODEL_CACHE_DIR}"
    local model_path=""

    # Check if model has a custom path mapping
    if [[ -n "${MODEL_PATH_MAP[$model_name]:-}" && -n "${MODEL_PATH_MAP[$model_name]}" ]]; then
        local local_path="${MODEL_PATH_MAP[$model_name]}"
        if [[ -d "$local_path" ]]; then
            echo "Using local model path from mapping: $local_path" >&2
            model_path="$local_path"
        else
            echo "Warning: Mapped path $local_path does not exist, falling back to download" >&2
        fi
    fi

    # If we have a valid path from mapping, return it
    if [[ -n "$model_path" && -d "$model_path" ]]; then
        printf '%s\n' "$model_path"
        return 0
    fi

    # Check default cache locations
    local hf_cache_name="${model_name//\//--}"
    local possible_paths=(
        "$cache_dir/$model_name"
        "$cache_dir/$hf_cache_name"
        "$HOME/.cache/huggingface/hub/models--$hf_cache_name"
    )

    for path in "${possible_paths[@]}"; do
        if [[ -d "$path" ]]; then
            echo "Found model in cache: $path" >&2
            printf '%s\n' "$path"
            return 0
        fi
    done

    # Model not found locally, download using huggingface-cli
    echo "Model $model_name not found locally. Downloading using huggingface-cli..." >&2
    if ! download_model "$model_name" "$cache_dir" "$FORCE_DOWNLOAD"; then
        echo "Error: Failed to download model $model_name" >&2
        return 1
    fi

    local downloaded_path="$cache_dir/$hf_cache_name"
    # Verify the downloaded model exists
    if [[ ! -d "$downloaded_path" ]]; then
        echo "Error: Downloaded model path is invalid: $downloaded_path" >&2
        return 1
    fi

    # Output only the path once
    printf '%s\n' "$downloaded_path"
    return 0
}

# Download a model using huggingface-cli
download_model() {
    local model_name="$1"
    local cache_dir="$2"
    local force="${3:-false}"

    # Check if huggingface-cli is available
    local hf_cli
    hf_cli=$(command -v hf || echo "")
    if [[ -z "$hf_cli" ]]; then
        echo "Error: huggingface-cli not found. Please install it with: pip install huggingface_hub[cli]" >&2
        exit 1
    fi

    # Create cache directory if it doesn't exist
    mkdir -p "$cache_dir"

    # Download model using huggingface-cli
    local hf_cache_name="${model_name//\//--}"
    local local_dir="$cache_dir/$hf_cache_name"

    # Check if model already exists
    if [[ -d "$local_dir" ]]; then
        if [[ "$force" == "true" ]]; then
            echo "⚠ Force downloading $model_name (overwriting existing model at $local_dir)" >&2
        else
            echo "✓ $model_name already exists at $local_dir (skipped)" >&2
            return 0
        fi
    fi

    echo "" >&2
    echo "============================================================" >&2
    echo "Downloading $model_name" >&2
    echo "Destination: $local_dir" >&2
    echo "============================================================" >&2

    if ! "$hf_cli" download "$model_name" --local-dir "$local_dir"; then
        echo "Error: Failed to download model $model_name" >&2
        exit 1
    fi

    echo "✓ Successfully downloaded $model_name to $local_dir" >&2
    echo "" >&2
}

# Get and validate model path
# Returns the model path if it exists, or exits with error if not found
get_and_validate_model_path() {
    local model_name="$1"
    local use_hf="${USE_HF_MODEL_PATH:-false}"

    local model_path
    if [[ "$use_hf" == "true" ]]; then
        # Use HuggingFace model path directly
        model_path="$model_name"
        if [[ -z "$model_path" ]]; then
            echo "Error: Model name is empty" >&2
            return 1
        fi
        echo "$model_path"
        return 0
    else
        # Get local model path
        # Capture only stdout (the model path), redirect stderr to /dev/null for clean capture
        if ! model_path=$(get_model_path "$model_name" "$MODEL_CACHE_DIR" 2>/dev/null); then
            echo "Error: Failed to get model path for: $model_name" >&2
            return 1
        fi
        # Take only the first line in case of any duplication
        model_path=$(echo "$model_path" | head -n1)

        # Check if model_path is empty
        if [[ -z "$model_path" ]]; then
            echo "Error: Model path is empty for: $model_name" >&2
            return 1
        fi

        # Check if model path exists
        if [[ ! -d "$model_path" ]]; then
            echo "Error: Model path does not exist or is invalid: $model_path" >&2
            echo "  Model: $model_name" >&2
            return 1
        fi

        # Output only once
        printf '%s\n' "$model_path"
        return 0
    fi
}

# Run a Python script with given arguments
run_script() {
    local script_path="$1"
    shift
    local script_args=("$@")

    # Extract output_dir from the last argument (passed separately)
    # The output_dir is passed as the last argument after all script args
    local output_dir=""
    local last_arg="${script_args[-1]}"

    # Check if last arg looks like a directory path (not a flag starting with --)
    if [[ "$last_arg" != --* ]] && [[ -n "$last_arg" ]]; then
        # Last argument is the output_dir, remove it from script_args
        output_dir="$last_arg"
        # Rebuild script_args without the last element
        local new_args=()
        local i=0
        while [[ $i -lt $((${#script_args[@]} - 1)) ]]; do
            new_args+=("${script_args[$i]}")
            i=$((i+1))
        done
        script_args=("${new_args[@]}")
    else
        # No separate output_dir provided, try to extract from --output-dir or --output-wav
        local i=0
        while [[ $i -lt ${#script_args[@]} ]]; do
            case "${script_args[$i]}" in
                --output-dir|--output-wav)
                    if [[ $((i+1)) -lt ${#script_args[@]} ]]; then
                        output_dir="${script_args[$((i+1))]}"
                        break
                    fi
                    ;;
            esac
            i=$((i+1))
        done
    fi

    # If still no output_dir, use a default
    if [[ -z "$output_dir" ]]; then
        output_dir="outputs"
    fi

    # Convert output_dir to absolute path if it's relative
    if [[ "$output_dir" != /* ]]; then
        # Relative path - make it relative to the script directory (dry_run)
        output_dir="$SCRIPT_DIR/$output_dir"
    fi

    # Ensure output directory exists
    mkdir -p "$output_dir"

    local script_full_path="$EXAMPLES_DIR/$script_path"
    if [[ ! -f "$script_full_path" ]]; then
        echo "Error: Script not found: $script_full_path" >&2
        exit 1
    fi

    # Add enforce-eager flag if enabled
    if [[ "$ENFORCE_EAGER" == "true" ]]; then
        script_args+=("--enforce-eager")
    fi

    local cmd=(python "$script_full_path" "${script_args[@]}")

    # Set up profiling environment variable if enabled
    if [[ "$ENABLE_PROFILE" == "true" ]]; then
        if [[ -n "$output_dir" ]]; then
            local profiler_path="$output_dir/profiler"
            mkdir -p "$profiler_path"
            export VLLM_TORCH_PROFILER_DIR="$profiler_path"
        else
            mkdir -p "profiler_output"
            export VLLM_TORCH_PROFILER_DIR="profiler_output"
        fi
        echo "Profiling enabled: VLLM_TORCH_PROFILER_DIR=$VLLM_TORCH_PROFILER_DIR" >&2
    fi

    # Set up debug logging if enabled
    if [[ "$ENABLE_DEBUG" == "true" ]]; then
        export VLLM_LOGGING_LEVEL="DEBUG"
        echo "Debug logging enabled: VLLM_LOGGING_LEVEL=DEBUG" >&2
    fi

    # Capture and apply ZE_AFFINITY_MASK from environment if set
    if [[ -n "${ZE_AFFINITY_MASK:-}" ]]; then
        echo "ZE_AFFINITY_MASK set to: $ZE_AFFINITY_MASK" >&2
    fi

    echo "Running: ${cmd[*]}" >&2

    # Set up log file path - always save to output_dir
    local log_file_path="$output_dir/test.log"
    echo "Log file: $log_file_path" >&2

    # Ensure the output directory exists (should already exist, but double-check)
    mkdir -p "$output_dir"

    # Write header to log file
    {
        echo "Command: ${cmd[*]}"
        echo "Working directory: $VLLM_OMNI_ROOT"
        echo "Timestamp: $(date -Iseconds)"
        echo "=================================================================================="
        echo ""
    } | tee "$log_file_path"

    # Set up signal handlers to catch interrupts and mark as failed
    local interrupted=false
    local python_pid_file
    python_pid_file=$(mktemp)

    # Handler for SIGINT (Ctrl+C) and SIGTERM
    # Only kill the Python process, not the entire script
    handle_interrupt() {
        interrupted=true
        echo "" | tee -a "$log_file_path"
        echo "INTERRUPTED: Process was interrupted (SIGINT/SIGTERM)" | tee -a "$log_file_path"
        # Read the Python PID from the temp file
        if [[ -f "$python_pid_file" ]]; then
            local python_pid
            python_pid=$(cat "$python_pid_file" 2>/dev/null) || true
            if [[ -n "$python_pid" ]] && kill -0 "$python_pid" 2>/dev/null; then
                # Kill the process and its children, but not the entire process group
                # Use pkill to kill the process tree
                pkill -P "$python_pid" 2>/dev/null || true
                kill -TERM "$python_pid" 2>/dev/null || true
                sleep 1
                pkill -9 -P "$python_pid" 2>/dev/null || true
                kill -KILL "$python_pid" 2>/dev/null || true
            fi
        fi
        # Don't exit - let the script continue to mark this test as failed
        # The trap handler should not cause the script to exit
    }

    # Trap signals - mark as interrupted
    # Use a function that doesn't kill the script
    # The trap must not exit - it should just mark as interrupted and kill the Python process
    # After handling, ignore the signal so the script doesn't exit
    trap 'handle_interrupt; :' INT TERM

    # Run command and capture output using tee
    # Use PIPESTATUS to get the actual command's exit code, not tee's
    local return_code=0
    local exit_reason=""

    cd "$VLLM_OMNI_ROOT" && {
        # Temporarily disable exit on error to capture exit codes properly
        # Save current state
        local old_e_state
        if [[ $- == *e* ]]; then
            old_e_state=1
            set +e
        else
            old_e_state=0
        fi

        # Run the command in background and capture its PID
        # This allows the trap handler to kill only the Python process
        "${cmd[@]}" 2>&1 | tee -a "$log_file_path" &
        local pipeline_pid=$!

        # Find the Python process PID (it's the parent of processes in the pipeline)
        # Wait a moment for the process to start
        sleep 0.2
        local python_pid
        # Try to find the Python process by looking for the script name
        python_pid=$(pgrep -f "python.*$(basename "$script_full_path")" | head -n1) || true
        if [[ -z "$python_pid" ]]; then
            # Fallback: get the first Python process that's a child of the pipeline
            python_pid=$(pgrep -P "$pipeline_pid" | head -n1) || true
        fi
        if [[ -n "$python_pid" ]]; then
            echo "$python_pid" > "$python_pid_file"
        fi

        # Wait for the pipeline to complete
        wait $pipeline_pid 2>/dev/null || true
        return_code=${PIPESTATUS[0]}

        # If interrupted was set, override return_code
        if [[ "$interrupted" == "true" ]]; then
            return_code=130
        fi

        # Clean up PID file
        rm -f "$python_pid_file"

        # Restore original state
        if [[ $old_e_state -eq 1 ]]; then
            set -e
        fi

        # Check if process was killed by a signal
        if [[ $return_code -gt 128 ]]; then
            local signal_num=$((return_code - 128))
            exit_reason="killed by signal $signal_num"
            case $signal_num in
                2) exit_reason="interrupted (SIGINT)" ;;
                9) exit_reason="killed (SIGKILL)" ;;
                11) exit_reason="segmentation fault (SIGSEGV)" ;;
                15) exit_reason="terminated (SIGTERM)" ;;
                *) exit_reason="killed by signal $signal_num" ;;
            esac
        elif [[ $return_code -ne 0 ]]; then
            exit_reason="exited with code $return_code"
        fi
    }

    # Clear trap
    trap - INT TERM

    # Write footer to log file
    {
        echo ""
        echo "=================================================================================="
        if [[ "$interrupted" == "true" ]]; then
            echo "Exit code: 130 (INTERRUPTED)"
            echo "Exit reason: Process was interrupted by user (SIGINT/SIGTERM)"
        elif [[ -n "$exit_reason" ]]; then
            echo "Exit code: $return_code"
            echo "Exit reason: $exit_reason"
        else
            echo "Exit code: $return_code"
        fi
        echo "Completed at: $(date -Iseconds)"
        if [[ $return_code -ne 0 ]] || [[ "$interrupted" == "true" ]]; then
            echo "ERROR: Script failed"
            if [[ "$interrupted" == "true" ]]; then
                echo "  Reason: Process was interrupted (human interrupt or timeout)"
            elif [[ -n "$exit_reason" ]]; then
                echo "  Reason: $exit_reason"
            fi
        fi
    } | tee -a "$log_file_path"

    # Mark as failed if interrupted or non-zero exit code
    if [[ "$interrupted" == "true" ]] || [[ $return_code -ne 0 ]]; then
        if [[ "$interrupted" == "true" ]]; then
            echo "Error: Script was interrupted (SIGINT/SIGTERM)" >&2
            return 130  # Standard exit code for SIGINT
        else
            echo "Error: Script failed with return code $return_code" >&2
            if [[ -n "$exit_reason" ]]; then
                echo "  Reason: $exit_reason" >&2
            fi
            return 1
        fi
    fi

    return 0
}

# Helper function to convert path to absolute
# Resolves relative paths relative to the script directory (dry_run), not current working directory
get_absolute_path() {
    local path="$1"
    if [[ "$path" == /* ]]; then
        # Already absolute
        echo "$path"
    else
        # Relative path - convert to absolute
        # Use realpath if available (handles non-existent paths with -m flag)
        if command -v realpath >/dev/null 2>&1; then
            # Use script directory as base if path doesn't start with ./
            if [[ "$path" != ./* ]] && [[ "$path" != /* ]]; then
                realpath -m "$SCRIPT_DIR/$path"
            else
                realpath -m "$path"
            fi
        else
            # Fallback: construct absolute path from script directory
            local base_dir="$SCRIPT_DIR"
            # Handle paths starting with ./
            path="${path#./}"
            # Remove leading ./ if present
            if [[ "$path" == ./* ]]; then
                path="${path#./}"
            fi
            # Construct absolute path
            if [[ "$base_dir" == "/" ]]; then
                echo "/$path"
            else
                echo "$base_dir/$path"
            fi
        fi
    fi
}

# Helper function to create test image
create_test_image() {
    local image_path="$1"
    local color="${2:-red}"
    python3 -c "
from PIL import Image
img = Image.new('RGB', (512, 512), color='$color')
img.save('$image_path')
"
}

# Test functions for each model

# Qwen3-Omni Models
test_qwen3_omni_30b() {
    local output_dir="${1:-outputs/qwen3_omni_30b}"
    local model_name="Qwen/Qwen3-Omni-30B-A3B-Instruct"

    echo ""
    echo "============================================================"
    echo "Testing Qwen/Qwen3-Omni-30B-A3B-Instruct"
    echo "============================================================"

    mkdir -p "$output_dir"
    local model_path
    model_path=$(get_and_validate_model_path "$model_name")

    local args=(
        "--model" "$model_path"
        "--query-type" "use_audio"
        "--stage-init-timeout" "300"
        "--output-wav" "$output_dir/output_audio.wav"
    )

    run_script "qwen3_omni/end2end.py" "${args[@]}" "$output_dir"
    echo "✓ Qwen3-Omni-30B test completed"
    echo ""
}

test_qwen2_5_omni_7b() {
    local output_dir="${1:-outputs/qwen2_5_omni_7b}"
    local model_name="Qwen/Qwen2.5-Omni-7B"

    echo ""
    echo "============================================================"
    echo "Testing Qwen/Qwen2.5-Omni-7B"
    echo "============================================================"

    mkdir -p "$output_dir"
    local model_path
    model_path=$(get_and_validate_model_path "$model_name")

    local args=(
        "--model" "$model_path"
        "--query-type" "use_mixed_modalities"
        "--output-wav" "$output_dir/output_audio.wav"
    )

    run_script "qwen2_5_omni/end2end.py" "${args[@]}" "$output_dir"
    echo "✓ Qwen2.5-Omni-7B test completed"
    echo ""
}

test_qwen2_5_omni_3b() {
    local output_dir="${1:-outputs/qwen2_5_omni_3b}"
    local model_name="Qwen/Qwen2.5-Omni-3B"

    echo ""
    echo "============================================================"
    echo "Testing Qwen/Qwen2.5-Omni-3B"
    echo "============================================================"

    mkdir -p "$output_dir"
    local model_path
    model_path=$(get_and_validate_model_path "$model_name")

    local args=(
        "--model" "$model_path"
        "--query-type" "use_mixed_modalities"
        "--output-wav" "$output_dir/output_audio.wav"
    )

    run_script "qwen2_5_omni/end2end.py" "${args[@]}" "$output_dir"
    echo "✓ Qwen2.5-Omni-3B test completed"
    echo ""
}

# BAGEL Model
test_bagel_7b_mot_text2img() {
    local output_dir="${1:-outputs/bagel_7b_mot}"
    local model_name="ByteDance-Seed/BAGEL-7B-MoT"

    echo ""
    echo "============================================================"
    echo "Testing ByteDance-Seed/BAGEL-7B-MoT"
    echo "============================================================"

    mkdir -p "$output_dir"
    local model_path
    model_path=$(get_and_validate_model_path "$model_name")

    local args=(
        "--model" "$model_path"
        "--prompts" "A cute cat"
        "--modality" "text2img"
        "--output-dir" "$output_dir"
    )

    run_script "bagel/end2end.py" "${args[@]}" "$output_dir"
    echo "✓ BAGEL-7B-MoT Text2Image test completed"
    echo ""
}

test_bagel_7b_mot_img2img() {
    local output_dir="${1:-outputs/bagel_7b_mot}"
    local model_name="ByteDance-Seed/BAGEL-7B-MoT"

    echo ""
    echo "============================================================"
    echo "Testing ByteDance-Seed/BAGEL-7B-MoT"
    echo "============================================================"

    mkdir -p "$output_dir"
    local model_path
    model_path=$(get_and_validate_model_path "$model_name")

    local args=(
        "--model" "$model_path"
        "--prompts" "Let the woman wear a blue dress"
        "--modality" "img2img"
        "--image-path" "$EXAMPLES_DIR/bagel/woman.png"
        "--output-dir" "$output_dir"
    )

    run_script "bagel/end2end.py" "${args[@]}" "$output_dir"
    echo "✓ BAGEL-7B-MoT Image2Image test completed"
    echo ""
}

# Qwen Image Models
test_qwen_image() {
    local output_dir="${1:-outputs/qwen_image}"
    local model_name="Qwen/Qwen-Image"

    echo ""
    echo "============================================================"
    echo "Testing Qwen/Qwen-Image"
    echo "============================================================"

    mkdir -p "$output_dir"
    local output_path="$output_dir/qwen_image_output.png"
    # Convert to absolute path
    output_path=$(get_absolute_path "$output_path")
    local model_path
    model_path=$(get_and_validate_model_path "$model_name")

    local args=(
        "--model" "$model_path"
        "--prompt" "a cup of coffee on the table"
        "--output" "$output_path"
        "--seed" "42"
        "--num-inference-steps" "50"
    )

    run_script "text_to_image/text_to_image.py" "${args[@]}" "$output_dir"
    echo "✓ Qwen-Image test completed"
    echo ""
}

test_qwen_image_2512() {
    local output_dir="${1:-outputs/qwen_image_2512}"
    local model_name="Qwen/Qwen-Image-2512"

    echo ""
    echo "============================================================"
    echo "Testing Qwen/Qwen-Image-2512"
    echo "============================================================"

    mkdir -p "$output_dir"
    local output_path="$output_dir/qwen_image_2512_output.png"
    # Convert to absolute path
    output_path=$(get_absolute_path "$output_path")
    local model_path
    model_path=$(get_and_validate_model_path "$model_name")

    local args=(
        "--model" "$model_path"
        "--prompt" "a cup of coffee on the table"
        "--output" "$output_path"
        "--seed" "42"
        "--num-inference-steps" "50"
    )

    run_script "text_to_image/text_to_image.py" "${args[@]}" "$output_dir"
    echo "✓ Qwen-Image-2512 test completed"
    echo ""
}

test_qwen_image_edit() {
    local output_dir="${1:-outputs/qwen_image_edit}"
    local model_name="Qwen/Qwen-Image-Edit"

    echo ""
    echo "============================================================"
    echo "Testing Qwen/Qwen-Image-Edit"
    echo "============================================================"

    mkdir -p "$output_dir"

    # Use qwen-bear.png from the examples directory
    local input_image="$EXAMPLES_DIR/image_to_image/qwen-bear.png"
    if [[ ! -f "$input_image" ]]; then
        echo "Error: Input image not found: $input_image" >&2
        echo "Please download it with: wget https://vllm-public-assets.s3.us-west-2.amazonaws.com/omni-assets/qwen-bear.png -O $input_image" >&2
        return 1
    fi

    local output_path="$output_dir/qwen_image_edit_output.png"
    # Convert to absolute path
    output_path=$(get_absolute_path "$output_path")
    local model_path
    if ! model_path=$(get_and_validate_model_path "$model_name"); then
        return 1
    fi

    # Validate model_path is not empty
    if [[ -z "$model_path" ]]; then
        echo "Error: Model path is empty for model: $model_name" >&2
        return 1
    fi

    local args=(
        "--model" "$model_path"
        "--image" "$input_image"
        "--prompt" "Make this image blue"
        "--output" "$output_path"
        "--seed" "0"
        "--num-inference-steps" "50"
    )

    if ! run_script "image_to_image/image_edit.py" "${args[@]}" "$output_dir"; then
        return 1
    fi
    echo "✓ Qwen-Image-Edit test completed"
    echo ""
}

test_qwen_image_edit_2509() {
    local output_dir="${1:-outputs/qwen_image_edit_2509}"
    local model_name="Qwen/Qwen-Image-Edit-2509"

    echo ""
    echo "============================================================"
    echo "Testing Qwen/Qwen-Image-Edit-2509"
    echo "============================================================"

    mkdir -p "$output_dir"
    create_test_image "$output_dir/test_input.png" "red"

    local output_path="$output_dir/qwen_image_edit_2509_output.png"
    # Convert to absolute path
    output_path=$(get_absolute_path "$output_path")
    local model_path
    model_path=$(get_and_validate_model_path "$model_name")

    local args=(
        "--model" "$model_path"
        "--image" "$output_dir/test_input.png"
        "--prompt" "Make this image blue"
        "--output" "$output_path"
        "--seed" "0"
        "--num-inference-steps" "50"
    )

    run_script "image_to_image/image_edit.py" "${args[@]}" "$output_dir"
    echo "✓ Qwen-Image-Edit-2509 test completed"
    echo ""
}

test_qwen_image_layered() {
    local output_dir="${1:-outputs/qwen_image_layered}"
    local model_name="Qwen/Qwen-Image-Layered"

    echo ""
    echo "============================================================"
    echo "Testing Qwen/Qwen-Image-Layered"
    echo "============================================================"

    mkdir -p "$output_dir"
    create_test_image "$output_dir/test_input.png" "red"

    local output_path="$output_dir/layered"
    # Convert to absolute path
    output_path=$(get_absolute_path "$output_path")
    local model_path
    model_path=$(get_and_validate_model_path "$model_name")

    local args=(
        "--model" "$model_path"
        "--image" "$output_dir/test_input.png"
        "--prompt" ""
        "--output" "$output_path"
        "--seed" "0"
        "--num-inference-steps" "50"
        "--layers" "4"
        "--color-format" "RGBA"
    )

    run_script "image_to_image/image_edit.py" "${args[@]}" "$output_dir"
    echo "✓ Qwen-Image-Layered test completed"
    echo ""
}

# Other Image Models
test_z_image_turbo() {
    local output_dir="${1:-outputs/z_image_turbo}"
    local model_name="Tongyi-MAI/Z-Image-Turbo"

    echo ""
    echo "============================================================"
    echo "Testing Tongyi-MAI/Z-Image-Turbo"
    echo "============================================================"

    mkdir -p "$output_dir"
    local output_path="$output_dir/z_image_turbo_output.png"
    # Convert to absolute path
    output_path=$(get_absolute_path "$output_path")
    local model_path
    model_path=$(get_and_validate_model_path "$model_name")

    local args=(
        "--model" "$model_path"
        "--prompt" "a cup of coffee on the table"
        "--output" "$output_path"
        "--seed" "42"
        "--num-inference-steps" "50"
    )

    run_script "text_to_image/text_to_image.py" "${args[@]}" "$output_dir"
    echo "✓ Z-Image-Turbo test completed"
    echo ""
}

test_glm_image() {
    local output_dir="${1:-outputs/glm_image}"
    local model_name="zai-org/GLM-Image"

    echo ""
    echo "============================================================"
    echo "Testing zai-org/GLM-Image"
    echo "============================================================"

    mkdir -p "$output_dir"
    local output_path="$output_dir/glm_image_output.png"
    # Convert to absolute path
    output_path=$(get_absolute_path "$output_path")
    local model_path
    model_path=$(get_and_validate_model_path "$model_name")

    local args=(
        "--model" "$model_path"
        "--prompt" "a cup of coffee on the table"
        "--output" "$output_path"
        "--seed" "42"
        "--num-inference-steps" "50"
    )

    run_script "text_to_image/text_to_image.py" "${args[@]}" "$output_dir"
    echo "✓ GLM-Image test completed"
    echo ""
}

test_ovis_image() {
    local output_dir="${1:-outputs/ovis_image}"
    local model_name="AIDC-AI/Ovis-Image-7B"

    echo ""
    echo "============================================================"
    echo "Testing AIDC-AI/Ovis-Image-7B"
    echo "============================================================"

    mkdir -p "$output_dir"
    local output_path="$output_dir/ovis_image_output.png"
    # Convert to absolute path
    output_path=$(get_absolute_path "$output_path")
    local model_path
    model_path=$(get_and_validate_model_path "$model_name")

    local args=(
        "--model" "$model_path"
        "--prompt" "a cup of coffee on the table"
        "--output" "$output_path"
        "--seed" "42"
        "--num-inference-steps" "50"
    )

    run_script "text_to_image/text_to_image.py" "${args[@]}" "$output_dir"
    echo "✓ Ovis-Image test completed"
    echo ""
}

test_longcat_image() {
    local output_dir="${1:-outputs/longcat_image}"
    local model_name="meituan-longcat/LongCat-Image"

    echo ""
    echo "============================================================"
    echo "Testing meituan-longcat/LongCat-Image"
    echo "============================================================"

    mkdir -p "$output_dir"
    local output_path="$output_dir/longcat_image_output.png"
    # Convert to absolute path
    output_path=$(get_absolute_path "$output_path")
    local model_path
    model_path=$(get_and_validate_model_path "$model_name")

    local args=(
        "--model" "$model_path"
        "--prompt" "a cup of coffee on the table"
        "--output" "$output_path"
        "--seed" "42"
        "--num-inference-steps" "50"
    )

    run_script "text_to_image/text_to_image.py" "${args[@]}" "$output_dir"
    echo "✓ LongCat-Image test completed"
    echo ""
}

test_longcat_image_edit() {
    local output_dir="${1:-outputs/longcat_image_edit}"
    local model_name="meituan-longcat/LongCat-Image-Edit"

    echo ""
    echo "============================================================"
    echo "Testing meituan-longcat/LongCat-Image-Edit"
    echo "============================================================"

    mkdir -p "$output_dir"
    create_test_image "$output_dir/test_input.png" "red"

    local output_path="$output_dir/longcat_image_edit_output.png"
    # Convert to absolute path
    output_path=$(get_absolute_path "$output_path")
    local model_path
    model_path=$(get_and_validate_model_path "$model_name")

    local args=(
        "--model" "$model_path"
        "--image" "$output_dir/test_input.png"
        "--prompt" "Make this image blue"
        "--output" "$output_path"
        "--seed" "0"
        "--num-inference-steps" "50"
    )

    run_script "image_to_image/image_edit.py" "${args[@]}" "$output_dir"
    echo "✓ LongCat-Image-Edit test completed"
    echo ""
}

test_hunyuan_image_3_0() {
    local output_dir="${1:-outputs/hunyuan_image_3_0}"
    local model_name="tencent/HunyuanImage-3.0"

    echo ""
    echo "============================================================"
    echo "Testing tencent/HunyuanImage-3.0"
    echo "============================================================"

    mkdir -p "$output_dir"
    local output_path="$output_dir/hunyuan_image_3_0_output.png"
    # Convert to absolute path
    output_path=$(get_absolute_path "$output_path")
    local model_path
    model_path=$(get_and_validate_model_path "$model_name")

    local args=(
        "--model" "$model_path"
        "--prompt" "a cup of coffee on the table"
        "--output" "$output_path"
        "--seed" "42"
        "--num-inference-steps" "50"
        "--tensor-parallel-size" "4"
    )

    run_script "text_to_image/text_to_image.py" "${args[@]}" "$output_dir"
    echo "✓ HunyuanImage-3.0 test completed"
    echo ""
}

# Stable Diffusion Models
test_stable_diffusion_3_5_medium() {
    local output_dir="${1:-outputs/stable_diffusion_3_5_medium}"
    local model_name="stabilityai/stable-diffusion-3.5-medium"

    echo ""
    echo "============================================================"
    echo "Testing stabilityai/stable-diffusion-3.5-medium"
    echo "============================================================"

    mkdir -p "$output_dir"
    local output_path="$output_dir/stable_diffusion_3_5_medium_output.png"
    # Convert to absolute path
    output_path=$(get_absolute_path "$output_path")
    local model_path
    model_path=$(get_and_validate_model_path "$model_name")

    local args=(
        "--model" "$model_path"
        "--prompt" "a cup of coffee on the table"
        "--output" "$output_path"
        "--seed" "42"
        "--num-inference-steps" "50"
    )

    run_script "text_to_image/text_to_image.py" "${args[@]}" "$output_dir"
    echo "✓ stable-diffusion-3.5-medium test completed"
    echo ""
}

# FLUX Models
test_flux_2_klein_4b() {
    local output_dir="${1:-outputs/flux_2_klein_4b}"
    local model_name="black-forest-labs/FLUX.2-klein-4B"

    echo ""
    echo "============================================================"
    echo "Testing black-forest-labs/FLUX.2-klein-4B"
    echo "============================================================"

    mkdir -p "$output_dir"
    local output_path="$output_dir/flux_2_klein_4b_output.png"
    # Convert to absolute path
    output_path=$(get_absolute_path "$output_path")
    local model_path
    model_path=$(get_and_validate_model_path "$model_name")

    local args=(
        "--model" "$model_path"
        "--prompt" "a cup of coffee on the table"
        "--output" "$output_path"
        "--seed" "42"
        "--num-inference-steps" "50"
    )

    run_script "text_to_image/text_to_image.py" "${args[@]}" "$output_dir"
    echo "✓ FLUX.2-klein-4B test completed"
    echo ""
}

test_flux_2_klein_9b() {
    local output_dir="${1:-outputs/flux_2_klein_9b}"
    local model_name="black-forest-labs/FLUX.2-klein-9B"

    echo ""
    echo "============================================================"
    echo "Testing black-forest-labs/FLUX.2-klein-9B"
    echo "============================================================"

    mkdir -p "$output_dir"
    local output_path="$output_dir/flux_2_klein_9b_output.png"
    # Convert to absolute path
    output_path=$(get_absolute_path "$output_path")
    local model_path
    model_path=$(get_and_validate_model_path "$model_name")

    local args=(
        "--model" "$model_path"
        "--prompt" "a cup of coffee on the table"
        "--output" "$output_path"
        "--seed" "42"
        "--num-inference-steps" "50"
    )

    run_script "text_to_image/text_to_image.py" "${args[@]}" "$output_dir"
    echo "✓ FLUX.2-klein-9B test completed"
    echo ""
}

test_flux_1_dev() {
    local output_dir="${1:-outputs/flux_1_dev}"
    local model_name="black-forest-labs/FLUX.1-dev"

    echo ""
    echo "============================================================"
    echo "Testing black-forest-labs/FLUX.1-dev"
    echo "============================================================"

    mkdir -p "$output_dir"
    local output_path="$output_dir/flux_1_dev_output.png"
    # Convert to absolute path
    output_path=$(get_absolute_path "$output_path")
    local model_path
    model_path=$(get_and_validate_model_path "$model_name")

    local args=(
        "--model" "$model_path"
        "--prompt" "a cup of coffee on the table"
        "--output" "$output_path"
        "--seed" "42"
        "--num-inference-steps" "50"
    )

    run_script "text_to_image/text_to_image.py" "${args[@]}" "$output_dir"
    echo "✓ FLUX.1-dev test completed"
    echo ""
}

# Video Models
test_wan2_2_t2v_a14b() {
    local output_dir="${1:-outputs/wan2_2_t2v_a14b}"
    local model_name="Wan-AI/Wan2.2-T2V-A14B-Diffusers"

    echo ""
    echo "============================================================"
    echo "Testing Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    echo "============================================================"

    mkdir -p "$output_dir"
    local output_path="$output_dir/wan2_2_t2v_a14b_output.mp4"
    # Convert to absolute path
    output_path=$(get_absolute_path "$output_path")
    local model_path
    model_path=$(get_and_validate_model_path "$model_name")

    local args=(
        "--model" "$model_path"
        "--prompt" "A serene lakeside sunrise with mist over the water."
        "--output" "$output_path"
        "--seed" "42"
        "--num-inference-steps" "40"
        "--num-frames" "81"
        "--tensor-parallel-size" "4"
    )

    run_script "text_to_video/text_to_video.py" "${args[@]}" "$output_dir"
    echo "✓ Wan2.2-T2V-A14B test completed"
    echo ""
}

test_wan2_2_ti2v_5b() {
    local output_dir="${1:-outputs/wan2_2_ti2v_5b}"
    local model_name="Wan-AI/Wan2.2-TI2V-5B-Diffusers"

    echo ""
    echo "============================================================"
    echo "Testing Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    echo "============================================================"

    mkdir -p "$output_dir"

    # Use cherry_blossom.jpg from the examples directory
    local input_image="$EXAMPLES_DIR/image_to_video/cherry_blossom.jpg"
    if [[ ! -f "$input_image" ]]; then
        echo "Error: Input image not found: $input_image" >&2
        echo "Please download it with: wget https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/cherry_blossom.jpg -O $input_image" >&2
        return 1
    fi

    local output_path="$output_dir/wan2_2_ti2v_5b_output.mp4"
    # Convert to absolute path
    output_path=$(get_absolute_path "$output_path")
    local model_path
    if ! model_path=$(get_and_validate_model_path "$model_name"); then
        return 1
    fi

    local args=(
        "--model" "$model_path"
        "--image" "$input_image"
        "--prompt" "A cat playing with yarn"
        "--output" "$output_path"
        "--seed" "42"
        "--num-inference-steps" "50"
        "--num-frames" "81"
    )

    if ! run_script "image_to_video/image_to_video.py" "${args[@]}" "$output_dir"; then
        return 1
    fi
    echo "✓ Wan2.2-TI2V-5B test completed"
    echo ""
}

test_wan2_2_i2v_a14b() {
    local output_dir="${1:-outputs/wan2_2_i2v_a14b}"
    local model_name="Wan-AI/Wan2.2-I2V-A14B-Diffusers"

    echo ""
    echo "============================================================"
    echo "Testing Wan-AI/Wan2.2-I2V-A14B-Diffusers"
    echo "============================================================"

    mkdir -p "$output_dir"

    # Use cherry_blossom.jpg from the examples directory
    local input_image="$EXAMPLES_DIR/image_to_video/cherry_blossom.jpg"
    if [[ ! -f "$input_image" ]]; then
        echo "Error: Input image not found: $input_image" >&2
        echo "Please download it with: wget https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/cherry_blossom.jpg -O $input_image" >&2
        return 1
    fi

    local output_path="$output_dir/wan2_2_i2v_a14b_output.mp4"
    # Convert to absolute path
    output_path=$(get_absolute_path "$output_path")
    local model_path
    if ! model_path=$(get_and_validate_model_path "$model_name"); then
        return 1
    fi

    local args=(
        "--model" "$model_path"
        "--image" "$input_image"
        "--prompt" "A cat playing with yarn"
        "--output" "$output_path"
        "--seed" "42"
        "--num-inference-steps" "50"
        "--num-frames" "81"
        "--tensor-parallel-size" "2"
    )

    if ! run_script "image_to_video/image_to_video.py" "${args[@]}" "$output_dir"; then
        return 1
    fi
    echo "✓ Wan2.2-I2V-A14B test completed"
    echo ""
}

# Audio Models
test_stable_audio_open() {
    local output_dir="${1:-outputs/stable_audio_open}"
    local model_name="stabilityai/stable-audio-open-1.0"

    echo ""
    echo "============================================================"
    echo "Testing stabilityai/stable-audio-open-1.0"
    echo "============================================================"

    mkdir -p "$output_dir"
    local output_path="$output_dir/stable_audio_open_output.wav"
    # Convert to absolute path
    output_path=$(get_absolute_path "$output_path")
    local model_path
    model_path=$(get_and_validate_model_path "$model_name")

    local args=(
        "--model" "$model_path"
        "--prompt" "The sound of a hammer hitting a wooden surface."
        "--output" "$output_path"
        "--seed" "42"
        "--num-inference-steps" "100"
        "--audio-length" "10.0"
    )

    run_script "text_to_audio/text_to_audio.py" "${args[@]}" "$output_dir"
    echo "✓ stable-audio-open-1.0 test completed"
    echo ""
}

# TTS Models
test_qwen3_tts_custom_voice() {
    local output_dir="${1:-outputs/qwen3_tts_custom_voice}"
    local model_name="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

    echo ""
    echo "============================================================"
    echo "Testing Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    echo "============================================================"

    mkdir -p "$output_dir"
    local model_path
    model_path=$(get_and_validate_model_path "$model_name")

    local args=(
        "--model" "$model_path"
        "--query-type" "CustomVoice"
        "--output-wav" "$output_dir/custom_voice.wav"
    )

    run_script "qwen3_tts/end2end.py" "${args[@]}" "$output_dir"
    echo "✓ Qwen3-TTS-CustomVoice test completed"
    echo ""
}

test_qwen3_tts_voice_design() {
    local output_dir="${1:-outputs/qwen3_tts_voice_design}"
    local model_name="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"

    echo ""
    echo "============================================================"
    echo "Testing Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    echo "============================================================"

    mkdir -p "$output_dir"
    local model_path
    model_path=$(get_and_validate_model_path "$model_name")

    local args=(
        "--model" "$model_path"
        "--query-type" "VoiceDesign"
        "--output-wav" "$output_dir/voice_design.wav"
    )

    run_script "qwen3_tts/end2end.py" "${args[@]}" "$output_dir"
    echo "✓ Qwen3-TTS-VoiceDesign test completed"
    echo ""
}

test_qwen3_tts_base() {
    local output_dir="${1:-outputs/qwen3_tts_base}"
    local model_name="Qwen/Qwen3-TTS-12Hz-0.6B-Base"

    echo ""
    echo "============================================================"
    echo "Testing Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    echo "============================================================"

    mkdir -p "$output_dir"
    local model_path
    model_path=$(get_and_validate_model_path "$model_name")

    local args=(
        "--model" "$model_path"
        "--query-type" "Base"
        "--output-wav" "$output_dir/base.wav"
    )

    run_script "qwen3_tts/end2end.py" "${args[@]}" "$output_dir"
    echo "✓ Qwen3-TTS-Base test completed"
    echo ""
}

# Generic test dispatcher
run_test() {
    local test_name="$1"
    local output_dir="$2"

    case "$test_name" in
        qwen3_omni_30b)
            test_qwen3_omni_30b "$output_dir"
            ;;
        qwen2_5_omni_7b)
            test_qwen2_5_omni_7b "$output_dir"
            ;;
        qwen2_5_omni_3b)
            test_qwen2_5_omni_3b "$output_dir"
            ;;
        bagel_7b_mot_text2img)
            test_bagel_7b_mot_text2img "$output_dir"
            ;;
        bagel_7b_mot_img2img)
            test_bagel_7b_mot_img2img "$output_dir"
            ;;
        qwen_image)
            test_qwen_image "$output_dir"
            ;;
        qwen_image_2512)
            test_qwen_image_2512 "$output_dir"
            ;;
        qwen_image_edit)
            test_qwen_image_edit "$output_dir"
            ;;
        qwen_image_edit_2509)
            test_qwen_image_edit_2509 "$output_dir"
            ;;
        qwen_image_layered)
            test_qwen_image_layered "$output_dir"
            ;;
        z_image_turbo)
            test_z_image_turbo "$output_dir"
            ;;
        glm_image)
            test_glm_image "$output_dir"
            ;;
        ovis_image)
            test_ovis_image "$output_dir"
            ;;
        longcat_image)
            test_longcat_image "$output_dir"
            ;;
        longcat_image_edit)
            test_longcat_image_edit "$output_dir"
            ;;
        hunyuan_image_3_0)
            test_hunyuan_image_3_0 "$output_dir"
            ;;
        stable_diffusion_3_5_medium)
            test_stable_diffusion_3_5_medium "$output_dir"
            ;;
        flux_2_klein_4b)
            test_flux_2_klein_4b "$output_dir"
            ;;
        flux_2_klein_9b)
            test_flux_2_klein_9b "$output_dir"
            ;;
        flux_1_dev)
            test_flux_1_dev "$output_dir"
            ;;
        wan2_2_t2v_a14b)
            test_wan2_2_t2v_a14b "$output_dir"
            ;;
        wan2_2_ti2v_5b)
            test_wan2_2_ti2v_5b "$output_dir"
            ;;
        wan2_2_i2v_a14b)
            test_wan2_2_i2v_a14b "$output_dir"
            ;;
        stable_audio_open)
            test_stable_audio_open "$output_dir"
            ;;
        qwen3_tts_custom_voice)
            test_qwen3_tts_custom_voice "$output_dir"
            ;;
        qwen3_tts_voice_design)
            test_qwen3_tts_voice_design "$output_dir"
            ;;
        qwen3_tts_base)
            test_qwen3_tts_base "$output_dir"
            ;;
        *)
            echo "Error: Unknown test: $test_name" >&2
            return 1
            ;;
    esac
}

# Download models
download_models() {
    local model_selection="$1"
    local cache_dir="${2:-$MODEL_CACHE_DIR}"
    local force="${3:-false}"

    mkdir -p "$cache_dir"

    local models_to_download=()
    if [[ "$model_selection" == "all" ]]; then
        for model in "${!MODEL_NAME_MAP[@]}"; do
            models_to_download+=("${MODEL_NAME_MAP[$model]}")
        done
        echo ""
        echo "============================================================"
        echo "Downloading ${#models_to_download[@]} models"
        echo "Cache directory: $cache_dir"
        echo "Force download: $force"
        echo "============================================================"
        echo ""
    else
        if [[ -z "${MODEL_NAME_MAP[$model_selection]:-}" ]]; then
            echo "Error: Unknown model: $model_selection" >&2
            exit 1
        fi
        models_to_download=("${MODEL_NAME_MAP[$model_selection]}")
        echo ""
        echo "============================================================"
        echo "Downloading model: ${models_to_download[0]}"
        echo "Cache directory: $cache_dir"
        echo "Force download: $force"
        echo "============================================================"
        echo ""
    fi

    local downloaded=()
    local skipped=()
    local failed=()

    local i=1
    for model_name in "${models_to_download[@]}"; do
        echo -n "[$i/${#models_to_download[@]}] "
        if download_model "$model_name" "$cache_dir" "$force"; then
            if [[ -d "$cache_dir/${model_name//\//--}" ]]; then
                downloaded+=("$model_name")
            else
                skipped+=("$model_name")
            fi
        else
            failed+=("$model_name")
        fi
        i=$((i+1))
    done

    # Print summary
    echo ""
    echo "============================================================"
    echo "Download Summary"
    echo "============================================================"
    echo "✓ Downloaded: ${#downloaded[@]}/${#models_to_download[@]}"
    echo "⊘ Skipped (already exists): ${#skipped[@]}/${#models_to_download[@]}"
    if [[ ${#failed[@]} -gt 0 ]]; then
        echo "✗ Failed: ${#failed[@]}/${#models_to_download[@]}"
    fi

    if [[ ${#downloaded[@]} -gt 0 ]]; then
        echo ""
        echo "Downloaded models:"
        for model in "${downloaded[@]}"; do
            echo "  ✓ $model"
        done
    fi

    if [[ ${#skipped[@]} -gt 0 ]]; then
        echo ""
        echo "Skipped models (already exist):"
        for model in "${skipped[@]}"; do
            echo "  ⊘ $model"
        done
    fi

    if [[ ${#failed[@]} -gt 0 ]]; then
        echo ""
        echo "Failed downloads:"
        for model in "${failed[@]}"; do
            echo "  ✗ $model"
        done
    fi
    echo "============================================================"
    echo ""
}

# Main function
main() {
    parse_args "$@"

    # Handle download-only mode
    if [[ "$DOWNLOAD_ONLY" == "true" ]]; then
        download_models "$MODEL" "$MODEL_CACHE_DIR" "$FORCE_DOWNLOAD"
        return 0
    fi

    # Normal test mode
    local summary_log_path="$OUTPUT_DIR/test_summary.log"
    mkdir -p "$OUTPUT_DIR"

    # Write header to summary log
    {
        echo "=================================================================================="
        echo "Test Summary Log"
        echo "=================================================================================="
        echo "Start time: $(date -Iseconds)"
        echo "Model: $MODEL"
        echo "Output directory: $OUTPUT_DIR"
        echo "Model cache directory: $MODEL_CACHE_DIR"
        echo "Enable profile: $ENABLE_PROFILE"
        echo "Enforce eager: $ENFORCE_EAGER"
        echo "Use HF model path: $USE_HF_MODEL_PATH"
        echo "Enable debug: $ENABLE_DEBUG"
        if [[ -n "${ZE_AFFINITY_MASK:-}" ]]; then
            echo "ZE_AFFINITY_MASK: $ZE_AFFINITY_MASK"
        fi
        echo "=================================================================================="
        echo ""
    } > "$summary_log_path"

    if [[ "$MODEL" == "all" ]]; then
        echo "Testing all models..." | tee -a "$summary_log_path"
        echo "" | tee -a "$summary_log_path"

        local passed=()
        local failed=()
        local total=${#ENABLED_MODELS[@]}
        local count=0

        for model_name in "${ENABLED_MODELS[@]}"; do
            count=$((count+1))
            local output_dir="$OUTPUT_DIR/$model_name"
            local test_start=$(date +%s)

            # Write to console and summary log
            echo "[$count/$total] Testing $model_name..." | tee -a "$summary_log_path"
            local start_time=$(date -Iseconds)
            echo "  Start time: $start_time" | tee -a "$summary_log_path"
            echo "  Output directory: $output_dir" | tee -a "$summary_log_path"

            # Run test - detailed output goes to stderr (console) and individual test.log
            # Only capture success/failure status
            # Temporarily disable exit on error to handle interrupts gracefully
            set +e
            if run_test "$model_name" "$output_dir"; then
                set -e
                local test_end=$(date +%s)
                local duration=$((test_end - test_start))
                local end_time=$(date -Iseconds)
                echo "  ✓ PASSED (Duration: ${duration}s)" | tee -a "$summary_log_path"
                echo "  End time: $end_time" | tee -a "$summary_log_path"
                echo "" | tee -a "$summary_log_path"
                passed+=("$model_name")
            else
                set -e
                local test_end=$(date +%s)
                local duration=$((test_end - test_start))
                local end_time=$(date -Iseconds)
                echo "  ✗ FAILED (Duration: ${duration}s)" | tee -a "$summary_log_path"
                echo "  End time: $end_time" | tee -a "$summary_log_path"
                echo "" | tee -a "$summary_log_path"
                failed+=("$model_name")
            fi
        done

        # Write final summary
        local end_time=$(date -Iseconds)
        {
            echo "=================================================================================="
            echo "Final Summary"
            echo "=================================================================================="
            echo "Total tests: $total"
            echo "Passed: ${#passed[@]}"
            echo "Failed: ${#failed[@]}"
            echo "End time: $end_time"
            echo ""

            if [[ ${#passed[@]} -gt 0 ]]; then
                echo "Passed tests:"
                for model in "${passed[@]}"; do
                    echo "  ✓ $model"
                done
                echo ""
            fi

            if [[ ${#failed[@]} -gt 0 ]]; then
                echo "Failed tests:"
                for model in "${failed[@]}"; do
                    echo "  ✗ $model"
                done
                echo ""
            fi
            echo "=================================================================================="
        } | tee -a "$summary_log_path"
    else
        # Single model test
        if [[ -z "${MODEL_NAME_MAP[$MODEL]:-}" ]]; then
            echo "Error: Unknown model: $MODEL" >&2
            echo "Available models: ${!MODEL_NAME_MAP[*]}" >&2
            exit 1
        fi

        local output_dir="$OUTPUT_DIR/$MODEL"
        local test_start=$(date +%s)

        echo "Testing $MODEL..." | tee -a "$summary_log_path"
        local start_time=$(date -Iseconds)
        echo "  Start time: $start_time" | tee -a "$summary_log_path"
        echo "  Output directory: $output_dir" | tee -a "$summary_log_path"

        # Run test - detailed output goes to stderr (console) and individual test.log
        if run_test "$MODEL" "$output_dir"; then
            local test_end=$(date +%s)
            local duration=$((test_end - test_start))
            local end_time=$(date -Iseconds)
            echo "  ✓ PASSED (Duration: ${duration}s)" | tee -a "$summary_log_path"
            echo "  End time: $end_time" | tee -a "$summary_log_path"
        else
            local test_end=$(date +%s)
            local duration=$((test_end - test_start))
            local end_time=$(date -Iseconds)
            echo "  ✗ FAILED (Duration: ${duration}s)" | tee -a "$summary_log_path"
            echo "  End time: $end_time" | tee -a "$summary_log_path"
            exit 1
        fi
    fi

    echo ""
    echo "============================================================"
    echo "Summary log: $summary_log_path"
    echo "============================================================"
    echo ""
}

# Run main function
main "$@"
