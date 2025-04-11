## Run training script 
### Run the venv
```bash
source ../../image-search/bin/activate
```

```bash
image-search\Scripts\activate
```

### Run training script 

```bash
python multitask_learning/MTL_training/train.py \
    --data_dir "/media/alexosama/data/youssefmohamed/image_search/images" \
    --img_size 128 \
    --styles_csv_path "/media/alexosama/data/youssefmohamed/image_search/datasets/styles.csv" \
    --dataset_name "botit" \ 
    --training_type "MTL-w1" \
    --class_weights_type "normal" \
    --num_epochs 1 \
    --batch_size 128 \
    --lr 0.001 \
    --initial_weights_config_path "/media/alexosama/data/youssefmohamed/image_search/multitask_learning/MTL_training/configs/initial_weights_config.json" \
    --save_model_path "/media/alexosama/data/youssefmohamed/image_search/multitask_learning/MTL_training/trained_models/fashion_classifier_2.pth"
```

```bash
python multitask_learning/MTL_training/train.py \
    --data_dir "/media/alexosama/data/youssefmohamed/image_search/botit_dataset" \
    --img_size 128 \
    --styles_csv_path "/media/alexosama/data/youssefmohamed/image_search/datasets/filtered_botit_dataset_cleaned.csv" \
    --dataset_name "botit" \
    --training_type "MTL-w1" \
    --class_weights_type "normal" \
    --num_epochs 1 \
    --batch_size 128 \
    --lr 0.001 \
    --initial_weights_config_path "/media/alexosama/data/youssefmohamed/image_search/multitask_learning/MTL_training/configs/initial_weights_config.json" \
    --save_model_path "/media/alexosama/data/youssefmohamed/image_search/multitask_learning/MTL_training/trained_models/MTL_fashion_classifier_1_botit_f1score.pth"
```

```bash
python multitask_learning/MTL_training/train.py \
    --data_dir "/media/alexosama/data/youssefmohamed/image_search/botit_dataset" \
    --img_size 128 \
    --styles_csv_path "/media/alexosama/data/youssefmohamed/image_search/datasets/filtered_botit_dataset_cleaned.csv" \
    --dataset_name "botit" \
    --training_type "MTL-DWBSTL" \
    --class_weights_type "NoN" \
    --num_epochs 20 \
    --batch_size 128 \
    --lr 0.001 \
    --initial_weights_config_path "/media/alexosama/data/youssefmohamed/image_search/multitask_learning/MTL_training/configs/botit_initial_weights_config.json" \
    --save_model_path "/media/alexosama/data/youssefmohamed/image_search/multitask_learning/MTL_training/trained_models/MTL-DWBSTL_fashion_classifier_1_botit_e20_f1score_nw.pth"
```

```bash
python multitask_learning/MTL_training/train.py \
    --data_dir "/media/alexosama/data/youssefmohamed/image_search/botit_dataset" \
    --img_size 128 \
    --styles_csv_path "/media/alexosama/data/youssefmohamed/image_search/datasets/filtered_botit_dataset_cleaned.csv" \
    --dataset_name "botit" \
    --training_type "MTL-w1" \
    --model_name "VGG16" \
    --class_weights_type "NoN" \
    --num_epochs 15 \
    --batch_size 128 \
    --lr 0.001 \
    --initial_weights_config_path "/media/alexosama/data/youssefmohamed/image_search/multitask_learning/MTL_training/configs/botit_initial_weights_config.json" \
    --save_model_path "/media/alexosama/data/youssefmohamed/image_search/multitask_learning/MTL_training/trained_models/MTL-w1_VGG_botit_e1_f1score_nw.pth"
```

#### xray
python multitask_learning/MTL_training/train.py \
    --data_dir "C:\Users\cse\Desktop\youssef\image_search\x-ray_data\images" \
    --img_size 128 \
    --styles_csv_path "C:\Users\cse\Desktop\youssef\image_search\x-ray_data\merged_deduplicated.csv" \
    --dataset_name "xray" \
    --training_type "MTL-DWBSTL" \
    --model_name "ResNet" \
    --class_weights_type "NoN" \
    --num_epochs 15 \
    --batch_size 128 \
    --lr 0.001 \
    --initial_weights_config_path "C:\Users\cse\Desktop\youssef\image_search\multitask_learning\MTL_training\configs\xray_initial_weights_resnet_config.json" \
    --save_model_path "C:\Users\cse\Desktop\youssef\image_search\multitask_learning\trained_models\MTL-DWBSTL_ResNet_xray_e1_f1score_nw.pth"

python multitask_learning/MTL_training/train.py \
    --data_dir "C:\Users\cse\Desktop\youssef\image_search\x-ray_data\images" \
    --img_size 128 \
    --styles_csv_path "C:\Users\cse\Desktop\youssef\image_search\x-ray_data\merged_deduplicated.csv" \
    --dataset_name "xray" \
    --training_type "MTL-w1" \
    --model_name "ResNet" \
    --class_weights_type "NoN" \
    --num_epochs 15 \
    --batch_size 128 \
    --lr 0.001 \
    --initial_weights_config_path "C:\Users\cse\Desktop\youssef\image_search\multitask_learning\MTL_training\configs\xray_initial_weights_cnnmodel1_config.json" \
    --save_model_path "C:\Users\cse\Desktop\youssef\image_search\multitask_learning\trained_models\MTL-w1_ResNet_xray_e1_f1score_nw.pth"

### Run testing script

```bash
python multitask_learning/MTL_training/test.py \
    --data_dir "/media/alexosama/data/youssefmohamed/image_search/images" \
    --styles_csv_path "/media/alexosama/data/youssefmohamed/image_search/datasets/styles.csv" \
    --model_path "/media/alexosama/data/youssefmohamed/image_search/multitask_learning/MTL_training/trained_models/fashion_classifier.pth" \
    --batch_size 128
    --img_size 128
```

```bash
python multitask_learning/MTL_training/test.py \
    --data_dir "/media/alexosama/data/youssefmohamed/image_search/botit_dataset" \
    --dataset_name "botit" \
    --styles_csv_path "/media/alexosama/data/youssefmohamed/image_search/datasets/filtered_botit_dataset_cleaned.csv" \
    --model_path "/media/alexosama/data/youssefmohamed/image_search/multitask_learning/MTL_training/trained_models/MTL-DWBSTL_fashion_classifier_1_botit_e20.pth" \
    --batch_size 128 \
    --img_size 128
```

```bash
python multitask_learning/MTL_training/test.py \
    --data_dir "C:/Users/cse/Desktop/youssef/image_search/x-ray_data/images" \
    --dataset_name "xray" \
    --model_name "ResNet" \
    --styles_csv_path "C:/Users/cse/Desktop/youssef/image_search/x-ray_data/merged_deduplicated.csv" \
    --model_path "C:\Users\cse\Desktop\youssef\image_search\multitask_learning\trained_models\MTL-DWBSTL_ResNet_xray_e1_f1score_nw.pth" \
    --batch_size 128 \
    --img_size 128
```

### Run image testing script

```bash
python multitask_learning/MTL_training/test_query.py \
    --model_path "/media/alexosama/data/youssefmohamed/image_search/multitask_learning/MTL_training/trained_models/fashion_classifier.pth" \
    --sentence "" \
```