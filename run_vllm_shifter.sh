#!/usr/bin/env bash
#SBATCH -J shifter-vllm-test
#SBATCH -C gpu
#SBATCH -N 2
#SBATCH -t 10
#SBATCH -o %x-%j.out

# Defaults
SHIFTER_IMAGE=${SHIFTER_IMAGE:-vllm/vllm-openai:v0.9.1}
DEBUG=${DEBUG:-}
GPUS_PER_NODE=4
NODES=1

usage() {
  echo "Usage: $0 [-i IMAGE] [-d] [-- <run_vllm.py args>]"
  echo "  -i IMAGE   Set container image (default: $SHIFTER_IMAGE)"
  echo "  -d         Enable NCCL debug mode"
  echo "  -g GPUS_PER_NODE  Number of GPUs per node (default: $GPUS_PER_NODE)"
  echo "  -n NODES          Number of nodes (default: $NODES)"
  echo "  --         Pass the rest as arguments to run_vllm.py"
  exit 1
}


# Parse options
while getopts ":i:dg:n:" opt; do
  case $opt in
    i) SHIFTER_IMAGE=$OPTARG ;;
    d) DEBUG="NCCL_DEBUG=INFO FI_LOG_LEVEL=debug" ;;
    g) GPUS_PER_NODE=$OPTARG ;;
    n) NODES=$OPTARG ;;
    \?) usage ;;
  esac
done
shift $((OPTIND-1))

# Export HF_HOME if set
if [ -n "$HF_HOME" ]; then
    export HF_HOME
    echo "Using HF_HOME: $HF_HOME"
fi

if [ -n "$HF_TOKEN" ]; then
    export HF_TOKEN
    echo "HF_TOKEN is set"
fi

if [[ $NODES -gt 1 ]]; then
  echo "[] Running on $NODES nodes"
  
  nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
  nodes_array=( $nodes )

  set -x
  # Ray Head Node
  echo "[] Starting ray head...."
  ray_head_node="${nodes_array[0]}"
  port=6379
  timeout_sec=120
  elapsed=0

  srun --nodes=1 --ntasks=1 --cpus-per-task=128 --gpus-per-task=4 -w $ray_head_node --unbuffered \
    shifter --image=$SHIFTER_IMAGE --module=gpu,nccl-plugin \
          bash -c "ray start --head --block" &

  while ! nc -z $ray_head_node $port; do
    sleep 5
    elapsed=$((elapsed+1))
    if [ $elapsed -ge $timeout_sec ]; then
        echo "ERROR: Ray head port $port did not open on $ray_head_node within $timeout_sec seconds."
        exit 1
    fi
  done
  echo "Ray head node is ready!"

  # Ray Worker Nodes
  worker_num=$(($NODES - 1)) #number of nodes other than the head node
  echo "<> Starting ${worker_num} ray workers..."
  for ((  i=1; i<=$worker_num; i++ )); do
    node_i=${nodes_array[$i]}
    echo "    - $i at $node_i"
    srun --nodes=1 --ntasks=1 --cpus-per-task=128 --gpus-per-task=4 -w $node_i --unbuffered \
        shifter --image=$SHIFTER_IMAGE --module=gpu,nccl-plugin \
          bash -c "ray start --address \"${ray_head_node}:6379\" --block" &
  done

  ray_init_timeout=300
  ray_cluster_size=$NODES
  interval=5
  active_nodes=0

  for (( elapsed=0; elapsed<ray_init_timeout; elapsed+=interval )); do
      active_nodes=$(shifter --image=$SHIFTER_IMAGE --module=gpu,nccl-plugin python3 -c \
          "import ray; ray.init(address='auto'); print(sum(node['Alive'] for node in ray.nodes()))")
      if [ "$active_nodes" -eq "$ray_cluster_size" ]; then
          echo "All ray workers are active and the ray cluster is initialized successfully."
          break
      fi
      echo "Wait for all ray workers to be active. $active_nodes/$ray_cluster_size is active"
      sleep ${interval}s
  done

  if [ "$active_nodes" -ne "$ray_cluster_size" ]; then
      echo "ERROR: Ray cluster did not initialize within timeout."
      exit 1
  fi

  shifter --image=$SHIFTER_IMAGE --module=gpu,nccl-plugin \
    bash -c "${DEBUG} python3 run_vllm.py $*"
  

  

else
  echo "[] Running on a single node"
  set -x
  srun \
      --nodes=$NODES \
      --ntasks-per-node=1 \
      --gpus-per-node=$GPUS_PER_NODE \
      --cpus-per-task=128 \
      shifter --image=$SHIFTER_IMAGE --module=gpu,nccl-plugin \
          bash -c "${DEBUG} python3 run_vllm.py $*"
fi

