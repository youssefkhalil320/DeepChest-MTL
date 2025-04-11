```bash
python "C:\Users\cse\Desktop\youssef\image_search\GradNorm_MTL\train.py" \
    --data_dir "C:\Users\cse\Desktop\youssef\image_search\x-ray_data\images" \
    --img_size 128 \
    --styles_csv_path "C:\Users\cse\Desktop\youssef\image_search\x-ray_data\merged_deduplicated.csv" \
    --dataset_name "xray" \
    --model_name "ResNet" \
    --num_epochs 15 \
    --batch_size 128 \
    --lr 0.001 \
    --save_model_path "C:\Users\cse\Desktop\youssef\image_search\GradNorm_MTL\trained_models\MTL_GradNorm_ResNet_xray_e1_f1score_nw.pth"
```

```bash
python "C:\Users\cse\Desktop\youssef\image_search\GradNorm_MTL\test.py" \
    --data_dir "C:\Users\cse\Desktop\youssef\image_search\x-ray_data\images" \
    --model_name "ResNet" \
    --dataset_name "xray" \
    --styles_csv_path "C:\Users\cse\Desktop\youssef\image_search\x-ray_data\merged_deduplicated.csv" \
    --model_path "C:\Users\cse\Desktop\youssef\image_search\GradNorm_MTL\trained_models\MTL_GradNorm_ResNet_xray_e1_f1score_nw.pth" \
    --batch_size 128 \
    --img_size 128
```