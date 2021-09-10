# Detectron2

Run Detectron2 example on SavviHub:
* Dataset mount
  1. Create a new dataset with a public S3 bucket directory `s3://savvihub-public-apne2/detectron2`.
  2. Mount the dataset to `/input` at the experiment create form.
* Start Command
  ```bash
  pip install -r detectron2/requirements.txt && python detectron2/main.py
  ```
