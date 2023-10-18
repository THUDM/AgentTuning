# WebArena

This evaluation code is adapted from [WebArena](https://github.com/web-arena-x/webarena).

## Install

```bash
# Python 3.10+
conda create -n webarena python=3.10; conda activate webarena
pip install -r requirements.txt
playwright install
pip install -e .
```

## Evaluation

1. Setup the standalone environment.

    Please check out [this page](environment_docker/README.md) for details.

2. Configurate the urls for each website.

    ```bash
    export SHOPPING="<your_shopping_site_domain>:7770"
    export SHOPPING_ADMIN="<your_e_commerce_cms_domain>:7780/admin"
    export REDDIT="<your_reddit_domain>:9999"
    export GITLAB="<your_gitlab_domain>:8023"
    export MAP="<your_map_domain>:3000"
    export WIKIPEDIA="<your_wikipedia_domain>:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
    export HOMEPAGE="<your_homepage_domain>:4399" # this is a placeholder
    ```

3. Obtain the auto-login cookies for all websites

    ```
    bash prepare.sh
    ```

4. Launch the evaluation

    ```bash
    bash eval-tgi.sh
    ```

    or `bash eval-gpt-3.5-turbo.sh` to run with GPT-3.5, or `bash eval-gpt-4.sh` to run with GPT-4.
    
    When evaluating, you need to modify the scripts to provide your OpenAI API key because the fuzzy match action requires access to OpenAI. When evaluating TGI models, you also need to modify the scripts to provide your TGI controller address.