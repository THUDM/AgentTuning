source eval.sh

# check if AGENT_CONFIG is set
if [ -z "$AGENT_CONFIG" ]; then
    echo "[ERROR] AGENT_CONFIG is not set"
    exit 1
fi

# check if SPLIT is set, if not, print warning and set to dev
if [ -z "$SPLIT" ]; then
    # print in yellow
    echo -e "\033[33m[WARNING] SPLIT is not set, set to dev\033[0m"
    SPLIT=dev
fi

OUTPUT_DIR="$OUTPUT_ROOT_DIR"
if [ -z "$OUTPUT_ROOT_DIR" ]; then
    TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
    AGENT_NAME=${AGENT_CONFIG##*/}
    AGENT_NAME=${AGENT_NAME%%\.yaml}
    echo $AGENT_NAME
    OUTPUT_DIR=outputs/0728-2/$AGENT_NAME
    OUTPUT_DIR=$(check_and_append_suffix "$OUTPUT_DIR")
    echo -e "\033[33m[INFO] OUTPUT_ROOT_DIR is not set, set to '$OUTPUT_DIR'\033[0m"
fi

if [ -z "$WORKERS" ]; then
    WORKERS=1
fi