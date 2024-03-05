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
Simply add the region name matching the expectations from the [upstream pipeline](https://github.com/SchmidtDSE/plastics-pipeline) to `regions.json` and execute the pipeline. This may require rebuilding the docker container.
