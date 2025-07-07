# A. Cloning and updating SARship-detect project

#### To clone the repository with ssh:

The following commands assume you have navigated into the local repo directory. 

```bash
git clone git@github.com:pmillitz/SARship-detect.git
```

To update the repos, issue a pull request:

- navigate into the local repo location

- fetch and merge the latest changes:

```bash
git pull
```

#### To see what’s changed before merging:

```bash
git fetch
git status
```

then you can inspect or manually merge:

```bash
git merge origin/main
```

#### To avoid overwriting local changes:

`Commit` or `stash` your changes first:

```bash
git add .
git commit -m "WIP: local changes"
```

or `stash` instead:

```bash
git stash
```

then pull:

```bash
git pull
```

if you stashed:

```bash
git stash pop
```

# B. Configuring the Jupyter notebook

1. Check that the `config.yaml` file looks like this:
   
   ```yaml
   #SARFish_root_directory: /mnt/h/SARFish/
   #Generated_root_directory: /mnt/h/SARFish/Generated
   
   SARFish_root_directory: /group/pmc044/data/SARFish/
   Generated_root_directory: /group/pmc044/data/Generated/
   
   product_type: SLC
   
   create_crop:
     correspondences_path: ./xView3_SLC_GRD_correspondences.csv
     annotations_path: ./SARFish_labels/validation/SLC/SLC_validation_labels.csv
     #arrays_path: /mnt/h/SARFish/Generated/SLC/train/arrays_raw/
     arrays_path: /group/pmc044/data/Generated/SLC/train/arrays_raw
     CREATE_CROP:
         #CropPath: /mnt/h/SARFish/Generated/SLC/train/crops/
         CropPath: /data/pmc044/data/Generated/SLC/train/crops
         CropSize: 96
         LabelConfidence: ['HIGH', 'MEDIUM']
         QuietMode: True
   
   batch_sar_processing:
     #input_dir: '/mnt/h/SARFish/Generated/SLC/train/crops/images/'
     input_dir: '/group/pmc044/data/Generated/SLC/train/crops/images/'
     #output_dir: '/mnt/h/SARFish/Generated/SLC/train/crops/images_proc/'
     output_dir: '/group/pmc044/data/Generated/SLC/train/crops/images_proc/'
     script_path: 'complex_scale_and_norm.py'
     nan_strategy: 'skip'
     epsilon: 1e-6
     verbose: False
     global_norm_params: [0.0, 70.0]  # or null for adaptive
     max_workers: 4  # or null for sequential
     file_pattern: '*.npy'
     skip_existing: True
     log_file: 'batch_sar_processing.log'  # or null to disable
   ```

2. Create symbolic links in the `working` directory to the generated data directories as follows:

```bash
ln -s /group/pmc044/data/Generated/SLC/train/crops/images images
ln -s /group/pmc044/data/Generated/SLC/train/crops/labels labels
```
