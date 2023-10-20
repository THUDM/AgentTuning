#!/bin/bash

function evaluate_directly() {
    echo "Evaluating Directly, Parameters: $*"
    python evaluate.py $*
}

# function evaluate_in_docker(image_name, evaluating_parameters)
function evaluate_in_docker() {
    DOCKER_IMAGE=$1
    shift
    echo "Evaluating in docker $DOCKER_IMAGE, Parameters: $*"
    docker run -it --rm --network host -v $(pwd):/root/workspace -w /root/workspace $DOCKER_IMAGE \
    bash -c "
            umask 0
            [ -f /root/.setup.sh ] && bash /root/.setup.sh
            python evaluate.py $*
    "
}

get_min() {
    if [ "$1" -lt "$2" ]; then
        echo "$1"
    else
        echo "$2"
    fi
}

check_and_append_suffix() {
    local dir="$1"
    local count=1
    local new_dir="${dir}"
    
    while [[ -d "${new_dir}" ]]; do
        new_dir="${dir}-$(printf '%04d' "${count}")"
        count=$((count + 1))
    done
    echo "${new_dir}"
}