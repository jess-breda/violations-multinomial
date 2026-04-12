- Fig S3 would be where the data is made but this is commented out, model is run on "new_trained_clean"
  data_type="new_trained_cleaned",
  data_path="/Users/jessbreda/Desktop/github/violations-multinomial/data",

        which comes from data_path + "/processed/all_animals_trained_threshold_cleaned.parquet"

  - this is made in the fig s3 notebook technically (but it's not actually made, must have made it during my FPO and just read from the same place)

- paths may need to be updated to be notebooks/publication/data/model_fits this is config.MODEL_FITS_PATH

- counts for initial raw df to trained final df need to be refined in a notebook. Will need some sort of data cleaning notebook that goes from raw updated v2 df to the trained df used in the publication and counts within, then the s3 fig can be just the actual fig and not the analyses.
  - maybe also some sort of eda NB
  - final script would be ideal

- fork from here and then delete?
