GHG Prep Pipeline
================================================================================
[Luigi](https://luigi.readthedocs.io/en/stable/)-based pipeline to sweep and select machine learning models which are used for the greenhouse gas emissions layer at [https://global-plastics-tool.org/](https://global-plastics-tool.org/).


<br>

Purpose
--------------------------------------------------------------------------------
Pipeline which executes pre-processing and model sweep / training before doing projections required by the GHG layer in [https://global-plastics-tool.org/](https://global-plastics-tool.org/). Note that, unlike the [larger pipeline repository](https://github.com/SchmidtDSE/plastics-pipeline), this only sweeps using ML methods before validating that error remains stable, reporting on the sweep for monitoring purposes.

<br>

Usage
--------------------------------------------------------------------------------
Most users can simply reference the output from the latest execution. That output is written to [https://global-plastics-tool.org/ghgpipeline.zip](https://global-plastics-tool.org/ghgpipeline.zip) and is publicly available under the [CC-BY-NC License](https://github.com/SchmidtDSE/plastics-pipeline/blob/main/LICENSE.md). That said, users may also leverage a local environment if desired. For common developer operations including adding regions or updating data, see the [cookbook](https://github.com/SchmidtDSE/plastics-ghg-pipeline/blob/main/COOKBOOK.md).

### Container Environment
A containerized Docker environment is available for execution. This will prepare outputs required for the [front-end tool](https://github.com/SchmidtDSE/plastics-prototype). See [cookbook](https://github.com/SchmidtDSE/plastics-ghg-pipeline/blob/main/COOKBOOK.md) for more details.

### Manual Environment
In addition to the Docker container, a manual environment can be established simply by running `pip install -r requirements.txt`. This assumes that sqlite3 is installed. Afterwards, simply run `bash build.sh`.

### Configuration
The configuration for the Luigi pipeline can be modified by providing a custom json file. See `task/job.json` for an example. Note that the pipeline, by default, uses random forest even though a full sweep is conducted because that approach tends to yield better avoidance of overfitting.

<br>

Tool
--------------------------------------------------------------------------------
Note that an interactive tool for this model is also available at [https://github.com/SchmidtDSE/plastics-prototype](https://github.com/SchmidtDSE/plastics-prototype).

<br>

Deployment
--------------------------------------------------------------------------------
This pipeline can be deployed by merging to the `deploy` branch of the repository, firing GitHub actions. This will cause the pipeline output files to be written to [https://global-plastics-tool.org/ghgpipeline.zip](https://global-plastics-tool.org/ghgpipeline.zip).

<br>

Local Environment
--------------------------------------------------------------------------------
Setup the local environment with `pip -r requirements.txt`.

<br>

Testing
--------------------------------------------------------------------------------
Some unit tests and other automated checks are available. The following is recommended:

```
$ pip install pycodestyle pyflakes nose2
$ pyflakes *.py
$ pycodestyle *.py
$ nose2
```

Note that unit tests and code quality checks are run in CI / CD.

<br>

Development Standards
--------------------------------------------------------------------------------
CI / CD should be passing before merges to `main` which is used to stage pipeline deployments and `deploy`. Where possible, please follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html). Please note that tests run as part of the pipeline itself and separate test files are not included. That said, developers should document which tasks are tests and expand these tests like typical unit tests as needed in the future. We allow lines to go to 100 characters. Please include docstrings where possible (optional for private members and tests, can assume dostrings are inherited).

<br>

Related Repositories
--------------------------------------------------------------------------------
See also [source code for the web-based tool](https://github.com/SchmidtDSE/plastics-prototype) running at [global-plastics-tool.org](https://global-plastics-tool.org) and [source code for "main" pipeline](https://github.com/SchmidtDSE/plastics-pipeline).

<br>

Open Source
--------------------------------------------------------------------------------
This project is released as open source (BSD and CC-BY-NC). See [LICENSE.md](https://github.com/SchmidtDSE/plastics-pipeline/blob/main/LICENSE.md) for further details. In addition to this, please note that this project uses the following open source:

 - [Luigi](https://luigi.readthedocs.io/en/stable/index.html) under the [Apache v2 License](https://github.com/spotify/luigi/blob/master/LICENSE).
 - [onnx](https://onnx.ai/) under the [Apache v2 License](https://github.com/onnx/onnx/blob/main/LICENSE).
 - [scikit-learn](https://scikit-learn.org/stable/) under the [BSD License](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING).
 - [sklearn-onnx](https://github.com/onnx/sklearn-onnx) under the [Apache v2 License](https://github.com/onnx/sklearn-onnx/blob/main/LICENSE).

The following are also potentially used as executables like from the command line but are not statically linked to code:

 - [Docker](https://docs.docker.com/engine/) under the [Apache v2 License](https://github.com/moby/moby/blob/master/LICENSE).
 - [Python 3.8](https://www.python.org/) under the [PSF License](https://docs.python.org/3/license.html).
