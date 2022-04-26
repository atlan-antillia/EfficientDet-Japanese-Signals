python model_inspect.py ^
  --runmode=saved_model ^
  --model_name=efficientdet-d0 ^
  --ckpt_path=./projects/Signals/models  ^
  --hparams="image_size=512x512,num_classes=5" ^
  --saved_model_dir=./projects/Signals/saved_model
 