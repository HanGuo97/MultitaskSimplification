# Data Preprocessing
Please follow the instructions from [Zhang et al. 2017](https://github.com/XingxingZhang/dress) for downloading the pre-processed dataset.
To build the .bin files please follow the instructions from [See et al. 2017](https://github.com/abisee/pointer-generator), or [here](https://github.com/abisee/cnn-dailymail).

# Evaluation Set-Up
* Please follow the instructions from [Zhang et al. 2017](https://github.com/XingxingZhang/dress) for setting up the evaluation system.
* FKGL implementations can be found [in this repo](https://github.com/mmautner/readability).
* Modify corresponding directories in `evaluation_utils/sentence_simplification.py`.
* Please note that evaluation metrics are calculated on corpus level.


# Dependencies
python 2.7  
tensorflow 1.4

# Usage
```bash
CUDA_VISIBLE_DEVICES="GPU_ID" python run.py \
    --mode "string" \
    --vocab_path "/path/to/vocab/file" \
    --train_data_dirs "/path/to/trainig/data_1,/path/to/trainig/data_2,/path/to/trainig/data_3" \
    --val_data_dir "/path/to/validation/data_1" \
    --decode_data_dir "/path/to/decode/data_1" \
    --eval_source_dir "/path/to/validation/data_1.source" \
    --eval_target_dir "/path/to/validation/data_1.target" \
    --max_enc_steps "int" --max_dec_steps "int" --batch_size "int" --steps_per_eval "int" \
    --log_root "/path/to/log/root/" --exp_name "string" [--autoMR] \
    --lr "float" --beam_size "int" --soft_sharing_coef "float"  --mixing_ratios "mr_1,mr_2"\
    --decode_ckpt_file "/path/to/ckpt" --decode_output_file "/path/to/file"

```
Pretrained models can be found [here](https://drive.google.com/file/d/1MJ6kq8nGfPcQaTZMreavkMET-BlG93Ij/view?usp=sharing).

# Citation
```
@inproceedings{guo2018dynamic,
    title = {Dynamic Multi-Level Multi-Task Learning for Sentence Simplification},
    author = {Han Guo and Ramakanth Pasunuru and Mohit Bansal},
    booktitle = {Proceedings of the 27th International Conference on Computational Linguistics (COLING 2018)},
    year = {2018}
}
```