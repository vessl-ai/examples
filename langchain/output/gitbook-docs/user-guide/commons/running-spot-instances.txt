# Running Spot Instances

VESSL supports Amazon EC2 Spot Instances on Amazon Elastic Kubernetes Service. Spot instances are attractive in terms of price and performance compared to on-demand instances, especially on stateless and fault-tolerant container runs.&#x20;

Be aware that spot instances are subject to interruptions. The claimed spot instances are suspended with 2 minutes of notice if the resource is needed elsewhere. Thus, saving and loading models for each epoch is highly recommended. Fortunately, most ML toolkits such as Fairseq and Detectron2, provide checkpointing which keeps the best-performing model. Refer to following documents to find more information about checkpointing:

* PyTorch: Saving and Loading Models
* TensorFlow: Save and Load Models

{% hint style="info" %}
Refer to example codes at VESSL GitHub repository.
{% endhint %}

### 1. Save Checkpoints

While training a model, you need to save the model periodically. The following PyTorch and Keras code compares validation accuracy and save the best performing model for each epoch. Note that the code keeps track of checkpoints so you can load the value as a `starch_epoch` value.&#x20;

{% tabs %}
{% tab title="PyTorch" %}
```python
import torch

def save_checkpoint(state, is_best, filename):
    if is_best:
        print("=> Saving a new best")
        torch.save(state, filename)
    else:
        print("=> Validation Accuracy did not improve")
        

for epoch in range(epochs):
    train(...)
    test_accuracy = 
    
    
    test_accuracy = torch.FloatTensor([test_accuracy]) 
    is_best = bool(test_accuracy.numpy() > best_accuracy.numpy())
    best_accuracy = torch.FloatTensor(
                max(test_accuracy.numpy(), best_accuracy.numpy()))
    save_checkpoint({
        'epoch': start_epoch + epoch + 1,
        'state_dict': model.state_dict(),
        'best_accuracy': best_accuracy,
    }, is_best, checkpoint_file_path)
```
{% endtab %}

{% tab title="Keras" %}
```python
from savvihub.keras import SavviHubCallback
from keras.callbacks import ModelCheckpoint
import os

checkpoint_path = os.path.join(args.checkpoint_path, 'checkpoints-{epoch:04d}.ckpt')
checkpoint_dir = os.path.dirname(checkpoint_path)

checkpoint_callback = ModelCheckpoint(
    checkpoint_path,
    monitor='val_accuracy',
    verbose=1,
    save_weights_only=True,
    mode='max',
    save_freq=args.save_model_freq,
)

# Compile model
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.save_weights(checkpoint_path.format(epoch=0))

model.fit(x_train, y_train,
          batch_size=args.batch_size,
          validation_data=(x_val, y_val),
          epochs=args.epochs,
          callbacks=[
              SavviHubCallback(
                  data_type='image',
                  validation_data=(x_val, y_val),
                  num_images=5,
                  start_epoch=start_epoch,
                  save_image=args.save_image,
              ),
              checkpoint_callback,
          ])
```
{% endtab %}
{% endtabs %}

### 2. Load Checkpoints

When spot instances are interrupted, the code is executed again from the beginning. To prevent this, you need to write a code that loads the saved checkpoint.

{% tabs %}
{% tab title="PyTorch" %}
```python
import torch
import os

def load_checkpoint(checkpoint_file_path):
    print(f"=> Loading checkpoint '{checkpoint_file_path}' ...")
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_file_path)
    else:
        checkpoint = torch.load(checkpoint_file_path, 
                        map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint.get('state_dict'))
    print(f"=> Loaded checkpoint (trained for {checkpoint.get('epoch')} epochs)")
    return checkpoint.get('epoch'), checkpoint.get('best_accuracy')


if os.path.exists(args.checkpoint_path) and os.path.isfile(checkpoint_file_path):
    start_epoch, best_accuracy = load_checkpoint(checkpoint_file_path)
else:
    print("=> No checkpoint has found! train from scratch")
    start_epoch, best_accuracy = 0, torch.FloatTensor([0])
    if not os.path.exists(args.checkpoint_path):
        print(f" [*] Make directories : {args.checkpoint_path}")
        os.makedirs(args.checkpoint_path)
```
{% endtab %}

{% tab title="Keras" %}
```python
import os
import tensorflow as tf

def parse_epoch(file_path):
    return int(os.path.splitext(os.path.basename(file_path))[0].split('-')[1])


checkpoint_path = os.path.join(args.checkpoint_path, 'checkpoints-{epoch:04d}.ckpt')
checkpoint_dir = os.path.dirname(checkpoint_path)
if os.path.exists(checkpoint_dir) and len(os.listdir(checkpoint_dir)) > 0:
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    print(f"=> Loading checkpoint '{latest}' ...")
    model.load_weights(latest)
    start_epoch = parse_epoch(latest)
    print(f'start_epoch:{start_epoch}')
else:
    start_epoch = 0
    if not os.path.exists(args.checkpoint_path):
        print(f" [*] Make directories : {args.checkpoint_path}")
        os.makedirs(args.checkpoint_path)
```
{% endtab %}
{% endtabs %}



The `start_epoch` value is a useful workaround to [logging metrics](broken-reference) to the __ VESSL server. Otherwise, the metrics graph might crash due to the spot instance interruption.

{% tabs %}
{% tab title="PyTorch" %}
```python
import savvihub

def train(...):
    ...
    savvihub.log(
        step=epoch+start_epoch+1, 
        row={'loss': loss.item()}
    )
```
{% endtab %}

{% tab title="Keras" %}
```python
from savvihub.keras import SavviHubCallback

model.fit(...,
    callbacks=[SavviHubCallback(
        ...,
        start_epoch=start_epoch,
        ...,
    )]
)
```
{% endtab %}
{% endtabs %}

### 3. Use the spot instance option

To use a spot instance on VESSL, click the **Use Spot Instance** checkbox. We also put the postfix \*.spot for every spot instance resource type. More resource types will be added in the future.\


![](https://files.gitbook.com/v0/b/gitbook-28427.appspot.com/o/assets%2F-MMUQrFFVCnMjEW\_\_WG9%2F-MT0h5l9X0FV77yaw7DD%2F-MT0l03ZzvBqhJLOMLnO%2Fspot\_instance.png?alt=media\&token=241f670c-c5d1-4e72-9317-8728a000f49f)
