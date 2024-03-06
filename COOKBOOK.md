# Cookbook
Guide for common developer operations in this repository.

<br>

## Executing the pipeline via Docker
One can run the pipeline through a containerized environment:

 - [Install Docker](https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-20-04)
 - Build the environment: `docker build -t dse/ghg_pipeline .`
 - Run the container: `docker run -it -d --name pipeline_run dse/ghg_pipeline bash`
 - Execute the pipeline: `docker exec -it pipeline_run bash clean_and_build.sh`
 - Zip the result: `docker exec -it pipeline_run zip ghg-pipeline.zip -r deploy`
 - Get the result: `docker cp pipeline_run:/workspace/ghg-pipeline.zip ghg-pipeline.zip`
 - Shutdown container: `docker stop pipeline_run`

Afterwards, `ghg-pipeline.zip` contains the results of the execution.

<br>

## Adding a region
Simply add the region name matching the expectations from the [upstream pipeline](https://github.com/SchmidtDSE/plastics-pipeline) to `regions.json` and execute the pipeline. This may require rebuilding the docker container. That said, this will require updating the input data as described below.

<br>

## Updating data
The input data are historical trade and socioeconomic (population / GDP) projections from the [base pipeline](https://github.com/SchmidtDSE/plastics-prototype) with trivial reorganization. In general, these data do not require updates upon changing the base pipeline unless those external projections or actuals change. Even so, these can be overidden from the [default file](https://global-plastics-tool.org/data/trade_inputs.csv) by changing `deploy/trade_inputs.csv`. For convienence, a [reference spreadsheet](https://docs.google.com/spreadsheets/d/1_M6FlgdmgPUIM4XNbT9Rb3DQMgM6SBru0Zi1qIyPGKI/edit#gid=203052437) for that data reorganization is available under CC-BY-NC.
