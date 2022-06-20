<h2>
EfficientDet Japanese Signals (Updated: 2022/06/20)
</h2>

This is a simple python example to train and detect Signals in Japan based on 
<a href="https://github.com/google/automl/tree/master/efficientdet">Google Brain AutoML efficientdet</a>.
<li>
Modified to use tensorflow 2.8.0 on Windows11. (2022/06/20)<br>
</li>
<li>
Modified to use the latest <a href="https://github.com/google/automl/tree/master/efficientdet">google/automl/efficientdet</a>.(2022/06/20)<br>
</li>


<h2>
1. Installing tensorflow on Windows11
</h2>
We use Python 3.8.10 to run tensoflow 2.8.0 on Windows11.<br>
<h3>1.1 Install Microsoft Visual Studio Community</h3>
Please install <a href="https://visualstudio.microsoft.com/ja/vs/community/">Microsoft Visual Studio Community</a>, 
which can be used to compile source code of 
<a href="https://github.com/cocodataset/cocoapi">cocoapi</a> for PythonAPI.<br>
<h3>1.2 Create a python virtualenv </h3>
Please run the following command to create a python virtualenv of name <b>py38-efficientdet</b>.
<pre>
>cd c:\
>python38\python.exe -m venv py38-efficientdet
>cd c:\py38-efficientdet
>./scripts/activate
</pre>
<h3>1.3 Create a working folder </h3>
Please create a working folder "c:\google" for your repository, and install the python packages.<br>

<pre>
>mkdir c:\google
>cd    c:\google
>pip install cython
>git clone https://github.com/cocodataset/cocoapi
>cd cocoapi/PythonAPI
</pre>
You have to modify extra_compiler_args in setup.py in the following way:
<pre>
   extra_compile_args=[]
</pre>
<pre>
>python setup.py build_ext install
</pre>

<br>

<br>
<h2>
2. Installing EfficientDet-Japanese-Signals
</h2>
<h3>2.1 clone EfficientDet-Japanese-Signals</h3>

Please clone EfficientDet-Japanese-Signals in the working folder <b>c:\google</b>.<br>
<pre>
>git clone  https://github.com/atlan-antillia/EfficientDet-Japanese-Signals.git<br>
</pre>
You can see the following folder <b>projects</b> in  EfficientDet-Japanese-Signals folder of the working folder.<br>

<pre>
EfficientDet-Japanese-Signals
└─projects
    └─Signals
        ├─eval
        ├─saved_model
        │  └─variables
        ├─test-dataset
        ├─test-dataset_outputs        
        ├─train
        └─valid
</pre>
<br>
<h3>2.2 Install python packages</h3>

Please run the following command to install python packages for this project.<br>
<pre>
>cd ./EfficientDet-Japanese-Signals
>pip install -r requirements.txt
</pre>
<h3>2.3 Workarounds for Windows</h3>
As you know or may not know, the efficientdet scripts of training a model and creating a saved_model do not 
run well on Windows environment in case of tensorflow 2.8.0(probably after the version 2.5.0) as shown below:. 
<pre>
INFO:tensorflow:Saving checkpoints for 0 into ./models\model.ckpt.
I0609 06:22:50.961521  3404 basic_session_run_hooks.py:634] Saving checkpoints for 0 into ./models\model.ckpt.
2022-06-09 06:22:52.780440: W tensorflow/core/framework/op_kernel.cc:1745] OP_REQUIRES failed at save_restore_v2_ops.cc:110 :
 NOT_FOUND: Failed to create a NewWriteableFile: ./models\model.ckpt-0_temp\part-00000-of-00001.data-00000-of-00001.tempstate8184773265919876648 :
</pre>

The real problem seems to happen in the original <b> save_restore_v2_ops.cc</b>. The simple workarounds to the issues are 
to modify the following tensorflow/python scripts in your virutalenv folder. 
<pre>
c:\py38-efficientdet\Lib\site-packages\tensorflow\python\training
 +- basic_session_run_hooks.py
 
634    logging.info("Saving checkpoints for %d into %s.", step, self._save_path)
635    ### workaround date="2022/06/18" os="Windows"
636    import platform
637    if platform.system() == "Windows":
638      self._save_path = self._save_path.replace("/", "\\")
639    #### workaround
</pre>

<pre>
c:\py38-efficientdet\Lib\site-packages\tensorflow\python\saved_model
 +- builder_impl.py

595    variables_path = saved_model_utils.get_variables_path(self._export_dir)
596    ### workaround date="2022/06/18" os="Windows" 
597    import platform
598    if platform.system() == "Windows":
599      variables_path = variables_path.replace("/", "\\")
600    ### workaround
</pre>

<br>

<h3>3. Inspect tfrecord</h3>
 Move to ./projects/Signals directory, and run the following bat file to inspect train/train.tfrecord:
<pre>
tfrecord_inspect.bat
</pre>
, which is the following:
<pre>
python ../../TFRecordInspector.py ^
  ./train/*.tfrecord ^
  ./label_map.pbtxt ^
  ./Inspector/train
</pre>
<br>
This will generate annotated images with bboxes and labels from the tfrecord, and cout the number of annotated objects in it.<br>

<b>TFRecordInspecotr: annotated images in train.tfrecord</b><br>
<img src="./asset/TFRecordInspector_train_annotated_images.png">
<br>
<br>
<b>TFRecordInspecotr: objects_count train.tfrecord</b><br>
<img src="./asset/TFRecordInspector_train_objects_count.png">
<br>
This bar graph shows that the number of the objects contained in train.tfrecord.
<br>
<br>
<br>
<h3>4. Downloading the pretrained-model efficientdet-d0</h3>
Please download an EfficientDet model chekcpoint file <b>efficientdet-d0.tar.gz</b>, and expand it in <b>EfficientDet-Japanese-Signals</b> folder.<br>
<br>
https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/efficientdet-d0.tar.gz
<br>
See: https://github.com/google/automl/tree/master/efficientdet<br>


<h3>5. Training Japanese Signals Model by using pretrained-model</h3>
Move to the ./projects/Signals directory, and run the following bat file to train signals efficientdet model:
<pre>
1_train.bat
</pre> 
, which is the following:
<pre>
python ../../ModelTrainer.py ^
  --mode=train_and_eval ^
  --train_file_pattern=./train/train.tfrecord  ^
  --val_file_pattern=./valid/valid.tfrecord ^
  --model_name=efficientdet-d0 ^
  --hparams="input_rand_hflip=False,image_size=512,num_classes=5,label_map=./label_map.yaml" ^
  --model_dir=./models ^
  --label_map_pbtxt=./label_map.pbtxt ^
  --eval_dir=./eval ^
  --ckpt=../../efficientdet-d0  ^
  --train_batch_size=4 ^
  --early_stopping=map ^
  --patience=10 ^
  --eval_batch_size=1 ^
  --eval_samples=200  ^
  --num_examples_per_epoch=400 ^
  --num_epochs=100
</pre>

<table style="border: 1px solid #000;">
<tr>
<td>
--mode</td><td>train_and_eval</td>
</tr>
<tr>
<td>
--train_file_pattern</td><td>./train/train.tfrecord</td>
</tr>
<tr>
<td>
--val_file_pattern</td><td>./valid/valid.tfrecord</td>
</tr>
<tr>
<td>
--model_name</td><td>efficientdet-d0</td>
</tr>
<tr><td>
--hparams</td><td>"input_rand_hflip=False,num_classes=5,label_map=./label_map.yaml"
</td></tr>
<tr>
<td>
--model_dir</td><td>./models</td>
</tr>
<tr><td>
--label_map_pbtxt</td><td>./label_map.pbtxt
</td></tr>

<tr><td>
--eval_dir</td><td>./eval
</td></tr>

<tr>
<td>
--ckpt</td><td>../../efficientdet-d0</td>
</tr>
<tr>
<td>
--train_batch_size</td><td>4</td>
</tr>
<tr>
<td>
--early_stopping</td><td>map</td>
</tr>
<tr>
<td>
--patience</td><td>10</td>
</tr>

<tr>
<td>
--eval_batch_size</td><td>1</td>
</tr>
<tr>
<td>
--eval_samples</td><td>200</td>
</tr>
<tr>
<td>
--num_examples_per_epoch</td><td>400</td>
</tr>
<tr>
<td>
--num_epochs</td><td>100</td>
</tr>
</table>
<br>
<br>
<b>label_map.yaml</b>
<pre>
1: 'Pedestrian_Signal_Blue'
2: 'Pedestrian_Signal_Red'
3: 'Traffic_Signal_Blue'
4: 'Traffic_Signal_Red'
5: 'Traffic_Signal_Yellow'
</pre>
<br>
<br>
<br>
<b>COCO meticss f and map</b><br>
<img src="./asset/coco_metrics_at_epoch47_tf2.8.0.png" width="1024" height="auto">
<br>
<br>
<b>Train losses at epoch</b><br>
<img src="./asset/train_losses_at_epoch47_tf2.8.0.png" width="1024" height="auto">
<br>
<br>

<b>COCO ap per class at epoch</b><br>
<img src="./asset/coco_ap_per_class_at_epoch47_tf2.8.0.png" width="1024" height="auto">
<br>

<h3>
6. Create a saved_model from the checkpoint
</h3>
 Please run the following bat file to create a saved model from a chekcpoint in models folder.
<pre>
2_create_saved_model.bat
</pre>
, which is the following:
<pre>
python ../../SavedModelCreator.py ^
  --runmode=saved_model ^
  --model_name=efficientdet-d0 ^
  --ckpt_path=./models  ^
  --hparams="image_size=512x512,num_classes=5" ^
  --saved_model_dir=./saved_model
</pre>

<table style="border: 1px solid #000;">
<tr>
<td>--runmode</td><td>saved_model</td>
</tr>

<tr>
<td>--model_name </td><td>efficientdet-d0 </td>
</tr>

<tr>
<td>--ckpt_path</td><td>./models</td>
</tr>

<tr>
<td>--hparams</td><td>"image_size=512x512,num_classes=5"</td>
</tr>

<tr>
<td>--saved_model_dir</td><td>./saved_model</td>
</tr>
</table>

<br>
<br>
<h3>
7. Inference Signals by using the saved_model
</h3>
 Please run the following bat file to infer the roadsigns by using the saved_model:
<pre>
</pre>
, which is the following:
<pre>
python ../../SavedModelInferencer.py ^
  --runmode=saved_model_infer ^
  --model_name=efficientdet-d0 ^
  --saved_model_dir=./saved_model ^
  --min_score_thresh=0.4 ^
  --hparams="label_map=./label_map.yaml" ^
  --input_image=./test_dataset/*.jpg ^
  --classes_file=./classes.txt ^
  --ground_truth_json=./test_dataset/annotation.json ^
  --output_image_dir=./test_dataset_outputs
</pre>

<table style="border: 1px solid #000;">
<tr>
<td>--runmode</td><td>saved_model_infer </td>
</tr>

<tr>
<td>--model_name</td><td>efficientdet-d0 </td>
</tr>

<tr>
<td>--saved_model_dir</td><td>./saved_model </td>
</tr>

<tr>
<td>--min_score_thresh</td><td>0.4</td>
</tr>

<tr>
<td>--hparams</td><td>"label_map=./label_map.yaml"</td>
</tr>

<tr>
<td>--input_image</td><td>./test_dataset/*.jpg</td>
</tr>
<tr>
<td>--classes_file</td><td>./classes.txt</td>
</tr>
<tr>
<td>--ground_truth_json</td><td>./realistic_test_dataset/annotation.json</td>
</tr>
<tr>
<td>--output_image_dir</td><td>./test_dataset_outputs</td>
</tr>
</table>
<br>
<h3>
8. Some detection results of Japanese Signals
</h3>

<img src="./projects/Signals/test_dataset_outputs/signals_1041.jpg" width="1280" height="auto"><br>
<a href="./projects/Signals/test_dataset_outputs/signals_1041.jpg_objects.csv">roadsigns1001.jpg_objects.csv</a><br>
<br>
<img src="./projects/Signals/test_dataset_outputs/signals_1042.jpg" width="1280" height="auto"><br>
<a  href="./projects/Signals/test_dataset_outputs/signals_1042.jpg_objects.csv">roadsigns1002.jpg_objects.csv</a><br>
<br>
<img src="./projects/Signals/test_dataset_outputs/signals_1043.jpg" width="1280" height="auto"><br>
<a  href="./projects/Signals/test_dataset_outputs/signals_1043.jpg_objects.csv">roadsigns1003.jpg_objects.csv</a><br>
<br>
<img src="./projects/Signals/test_dataset_outputs/signals_1044.jpg" width="1280" height="auto"><br>
<a  href="./projects/Signals/test_dataset_outputs/signals_1044.jpg_objects.csv">roadsigns1004.jpg_objects.csv</a><br>
<br>
<img src="./projects/Signals/test_dataset_outputs/signals_1045.jpg" width="1280" height="auto"><br>
<a  href="./projects/Signals/test_dataset_outputs/signals_1045.jpg_objects.csv">roadsigns1005.jpg_objects.csv</a><br>
<br>
<img src="./projects/Signals/test_dataset_outputs/signals_1046.jpg" width="1280" height="auto"><br>
<a  href="./projects/Signals/test_dataset_outputs/signals_1046.jpg_objects.csv">roadsigns1006.jpg_objects.csv</a><br>
<br>
<img src="./projects/Signals/test_dataset_outputs/signals_1047.jpg" width="1280" height="auto"><br>
<a  href="./projects/Signals/test_dataset_outputs/signals_1047.jpg_objects.csv">roadsigns1007.jpg_objects.csv</a><br>
<br>
<img src="./projects/Signals/test_dataset_outputs/signals_1048.jpg" width="1280" height="auto"><br>
<a  href="./projects/Signals/test_dataset_outputs/signals_1048.jpg_objects.csv">roadsigns1008.jpg_objects.csv</a><br>
<br>
<img src="./projects/Signals/test_dataset_outputs/signals_1049.jpg" width="1280" height="auto"><br>
<a  href="./projects/Signals/test_dataset_outputs/signals_1049.jpg_objects.csv">roadsigns1009.jpg_objects.csv</a><br>
<br>
<img src="./projects/Signals/test_dataset_outputs/signals_1050.jpg" width="1280" height="auto"><br>
<a  href="./projects/Signals/test_dataset_outputs/signals_1050.jpg_objects.csv">roadsigns1010.jpg_objects.csv</a><br>
<br>

<h3>9. COCO metrics of inference result</h3>
The 3_inference.bat computes also the COCO metrics(f, map, mar) to the <b>test_dataset</b> as shown below:<br>

<a href="./projects/Signals/test_dataset_outputs/prediction_f_map_mar.csv">prediction_f_map_mar.csv</a>

<br>
<img src="./asset/coco_metrics_console_test_dataset_at_epoch47_tf2.8.0.png" width="740" height="auto"><br>
