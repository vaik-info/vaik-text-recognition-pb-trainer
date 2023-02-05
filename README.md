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