# Defaults
MODE        ?= single
MODEL       ?=
PP_SIZE     ?= 
BACKEND     ?= shifter
IMAGE       ?= vllm/vllm-openai:v0.9.1
HF_HOME     ?= $(SCRATCH)/huggingface/

# Compose optional arguments
MODEL_ARG   := $(if $(MODEL),--model $(MODEL),)
PP_ARG      := $(if $(PP_SIZE),--pipeline-parallel-size $(PP_SIZE),)
IMAGE_ARG   := $(if $(IMAGE),-i $(IMAGE),)
HF_HOME_ARG := HF_HOME=$(HF_HOME)

# Select runner
ifeq ($(BACKEND),podman)
	RUNNER=$(error Error: 'podman' backend is currently disabled)
else
    RUNNER=./run_vllm_shifter.sh
endif

.PHONY: helprun single tp dp pp_tp

help:
	@echo ""
	@echo "vLLM Makefile Usage"
	@echo "-------------------"
	@echo "Targets:"
	@echo "  make run         # Run with user-specified MODE (see below)"
	@echo "  make single      # Run single-GPU mode"
	@echo "  make tp          # Run tensor parallel mode"
	@echo "  make dp          # (currently not working) Data parallel mode"
	@echo "  make pp_tp       # Run pipeline+tensor parallel mode"
	@echo ""
	@echo "Variables (can be set on command line):"
	@echo "  MODE=single|tp|dp|pp_tp         # Selects the vLLM mode (default: single)"
	@echo "  MODEL=repo/model                # Override the model for the run"
	@echo "  PP_SIZE=2                       # Override pipeline parallel size"
	@echo "  IMAGE=repo/image:tag            # Set the container image"
	@echo "  BACKEND=shifter|podman          # Select container backend (default: shifter)"
	@echo "  HF_HOME=/path/to/hf/cache       # Hugging Face cache directory (default: \$$SCRATCH/tmp)"
	@echo ""
	@echo "Examples:"
	@echo "  make tp"
	@echo "  make pp_tp PP_SIZE=2"
	@echo "  make tp IMAGE=myrepo/myimg:latest BACKEND=podman"
	@echo "  make run MODE=dp MODEL=facebook/opt-350m"
	@echo "  make tp HF_HOME=/my/hf/cache"
	@echo ""
	@echo "All variables can be overridden on the command line as shown above."
	@echo "For more details, see the README or run make help."

run:
	$(HF_HOME_ARG) $(RUNNER) $(IMAGE_ARG) -- --mode $(MODE) $(MODEL_ARG) $(PP_ARG)

single:
	$(HF_HOME_ARG) $(RUNNER) $(IMAGE_ARG) -g 1 -- --mode single $(MODEL_ARG)

tp:
	$(HF_HOME_ARG) $(RUNNER) $(IMAGE_ARG) -- --mode tp $(MODEL_ARG)

dp:
	@echo "Error: 'dp' (data parallel) mode is currently not working."
	@exit 1

pp_tp:
	$(HF_HOME_ARG) $(RUNNER) $(IMAGE_ARG) -n $(PP_SIZE) -- --mode pp_tp $(MODEL_ARG) $(PP_ARG)
