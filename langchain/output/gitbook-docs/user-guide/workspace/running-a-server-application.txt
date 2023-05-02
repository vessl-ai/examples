# Running a Server Application

### Port Configuration

Under **Port** setting of **Create New Workspace** page, you can expose a server application running in a VESSL workspace instance on a configured port.&#x20;

![](<../../.gitbook/assets/image (129).png>)

{% hint style="warning" %}
Before editing your port settings, you should stop your workspace.
{% endhint %}

![](<../../.gitbook/assets/image (235).png>)

### Running a Server Application

You can run a simple server application like the following Python file server.&#x20;

```
vessl@workspace-9fph3e8n3arx-0:~$ ls
mnist

vessl@workspace-9fph3e8n3arx-0:~$ ls mnist
test.csv  train.csv

vessl@workspace-9fph3e8n3arx-0:~$ python -m http.server 8080
Serving HTTP on 0.0.0.0 port 8080 (http://0.0.0.0:8080/) ...
```

You can access the running server application by clicking the port number under **METADATA**.

![](<../../.gitbook/assets/image (91).png>)

![](<../../.gitbook/assets/image (110).png>)
