# vaik-text-recoglition-pb-trainer

Train text recoglition pb model

## train_pb.py

### Usage

```shell
pip install -r requirements.txt
python train_pb.py --train_font_dir_path ~/vaik-text-recognition-pb-trainer/data/train_font/font \
                --valid_font_dir_path ~/vaik-text-recognition-pb-trainer/.venv/lib/python3.9/site-packages/vaik_text_generator/fonts \
                --char_json_path ~/vaik-text-recognition-pb-trainer/data/jpn_character.json \
                --classes_json_path ~/vaik-text-recognition-pb-trainer/data/number_plate_address.json \
                --model_type simple_conv_model \
                --epochs 1000 \
                --step_size 5000 \
                --batch_size 16 \
                --test_max_sample 100 \
                --image_height 96 \
                --output_dir_path ~/.vaik_text_recognition_pb_trainer/output_model
```

- train_font_dir_path & valid_font_dir_path

```shell
.
├── one.ttf
├── two.otf
・・・

```

### Output

![Screenshot from 2023-02-05 11-50-51](https://user-images.githubusercontent.com/116471878/216799078-9b26deb4-2b73-4ffb-b957-a4bd6b91e983.png)

![Screenshot from 2023-02-05 11-51-07](https://user-images.githubusercontent.com/116471878/216799082-20c838d7-e221-4172-b6cb-12676a3801e9.png)

-----

## export_fixed_model.py

### Usage

```shell
python export_fixed_model.py --input_weight_path ~/.vaik_text_recognition_pb_trainer/output_model/2023-02-21-23-35-54/step-5000_batch-16_epoch-35_loss_0.3708_val_loss_0.1772/step-5000_batch-16_epoch-35_loss_0.3708_val_loss_0.1772 \
                --char_json_path ~/vaik-text-recognition-pb-trainer/data/jpn_character.json \
                --model_type simple_conv_model \
                --image_height 96 \
                --image_width 512 \
                --output_dir_path ~/.vaik_text_recognition_pb_trainer/output_fixed_model
```

### Output

![Screenshot from 2023-02-22 17-47-30](https://user-images.githubusercontent.com/116471878/220569300-4ad933da-5793-4e86-9253-cbfacafea3c4.png)

-----

## dump_dataset.py

### Usage

```shell
python dump_dataset.py --train_font_dir_path ~/.vaik_text_recognition_pb_trainer/output_model/2023-02-21-23-35-54/step-5000_batch-16_epoch-35_loss_0.3708_val_loss_0.1772/step-5000_batch-16_epoch-35_loss_0.3708_val_loss_0.1772 \
                --char_json_path ~/vaik-text-recognition-pb-trainer/data/jpn_character.json \
                --model_type simple_conv_model \
                --classes_json_path ~/vaik-text-recognition-pb-trainer/data/number_plate_address.json \
                --image_height 96 \
                --image_width 512 \
                --sample_num 25000 \
                --output_dir_path ~/.vaik_text_recognition_pb_trainer/dump_dataset
```

### Output

![Screenshot from 2023-02-22 17-53-55](https://user-images.githubusercontent.com/116471878/220570348-36ea8bf5-013c-4fa3-bf02-c39f3b06a9fa.png)
