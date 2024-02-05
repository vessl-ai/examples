import os
import random 
import vessl

epochs = int(os.environ['epochs'])
lr = int(os.environ['lr'])
offset = random.random() / 5

for epoch in range(2, epochs):
    acc = 1 - 2**-epoch - random.random() / epoch - offset
    loss = 2**-epoch + random.random() / epoch + offset
    print(f"epoch={epoch}, accuracy={acc}, loss={loss}")
    vessl.log({"accuracy": acc, "loss": loss})
