python model_inspect.py ^
  --runmode=saved_model_infer ^
  --model_name=efficientdet-d0 ^
  --saved_model_dir=./projects/Signals/saved_model ^
  --min_score_thresh=0.2 ^
  --hparams="num_classes=5,label_map=./projects/Signals/label_map.yaml" ^
  --input_image=./projects/Signals/test_dataset/*.jpg ^
  --output_image_dir=./projects/Signals/test_dataset_outputs