# prad_dpai_survival
This is the DPAI models and training loop implementation for *"Comparison of Digital Histology AI Models with Low-Dimensional Genomic and Clinical Models in Survival Modeling for Prostate Cancer"*.


![Alt text for screen readers](./imgs/dpai_diagram.jpg "Optional tooltip/title")

<!--  -->

## Prerequisites


```text
torch 2.2.1
scikit-learn 1.3.0
scikit-survival 0.22.2
openslide 3.4.1
pandas 2.1.4
```

## Data Sets Prep

We summarize the setup steps for the data directory in order to run this DPAI model suite for any cohort of your choosing.

First, establish a parent directory for datasets, ``ALL_DATA_PAR_DIR``, as well as project name for your cohort, `COHORT_NAME`.  The skeleton and key `.csv` files for this data directory are created in `ALL_DATA_PAR_DIR = ./data` and `COHORT_NAME = tcga_prad` in this repository. For the following data sets, their root directory will be: `ALL_DATA_PAR_DIR/COHORT_NAME`

1. WSIs
    * Download your project's ```.svs``` WSI files and place in folder: ```wsi_files```
2. Survival / Progression Labels
    * Create file following the structure of `tabular_data/clindata/outcomes.csv`, which is the progression data for the TCGA PRAD cohort.
3. Clinical Data
    * Create data set with `case_id` column and any other clinical features of interest.  An example file is given at `tabular_data/clindata/tcga_prad_clin_imputed.csv`, which comes from [NCI Genomic Data Commons](https://portal.gdc.cancer.gov/). The given file includes preprocessing and KNN imputation for some missing clinical fields.
4. Bulk RNAseq
    * Our bulk RNAseq data comes directly from [NCI Genomic Data Commons](https://portal.gdc.cancer.gov/). We use transcripts per million (TPM), and apply $log_2(x+1)$ transformation followed by z-score normalization during training.  Processed omics file can be stored in `tabular_data/genomicdata`. If intending to run late fusion with genomic variables, be sure to concatenate them to the clinical data set, `tabular_data/clindata/tcga_prad_clin_imputed.csv`.
5. Genomic Pathways
    * A CSV containing the [Hallmark Pathways](https://www.gsea-msigdb.org/gsea/msigdb/collections.jsp) as well as the 5-gene PRAD signature is located at `tabular_data/genomicdata/pathways/hallmark_pathways_wprad.csv`. You may create your own custom pathway file and include here.
6. Batch Information
    * To examine whether performance is affected based on technical batches in your data, you may include a `batch_id` column in a file matching the structure and name of `tabular_data/metadata/slide_metadata.csv`.
7. Patch Feature Extraction Models
    * If using `UNI`, you must [download the model weights](https://huggingface.co/MahmoodLab/UNI). We keep the model checkpoint at `ALL_DATA_PAR_DIR/pretrained_models/uni/checkpointzip`. If you use an alternative location, please update `UNI_CKPT_DIR` constant in `modules/modeling/proj_constants.py`.

## Run Experiments with Pytorch Lightning

1. Check and Update `modules/modeling/proj_constants.py`

    This file sets key global parameters for the experiments, such as:
    * `BATCHES_IDX_EVEN`: indicates which batch IDs should be partitioned together for CV folds.
    * `ALL_DATA_PAR_DIR`: source directory of all project data files.
    * `PATCH_SIZE` and `MAGNIF`: sizing and resolution of patches.
    * `ENCODER`: feature extractor for patches.
    * `RUN_PREPROC`: boolean indicating whether to deploy shell script/s in `modules/preprocessing/shell_deployers/.` that will preprocess the fixed inputs for each DPAI model.

2. Setup experiments of interest and run `modules/modeling/run_models.py`

    This file includes high level deployment function (of each DPAI model) for a Pytorch Lightning cross-validated training loop.  Each deployment function provides at least the option to designate late-fusion clinical variables list and to specify either *batch stratified* or *batch isolated* training strategy. Examples are provided in the `__main__` guard of the script.


## Acknowledgements

The following public code repositories were very helpful for compiling our codebase:

* [CLAM](https://github.com/mahmoodlab/CLAM)
* [HVTSurv](https://github.com/szc19990412/HVTSurv)
* [CMTA](https://github.com/FT-ZHOU-ZZZ/CMTA)