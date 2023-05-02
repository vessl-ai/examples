# Getting Started

### Overview

To get started,&#x20;

1. Sign in to [https://www.vessl.ai](https://www.vessl.ai).&#x20;
2. In your terminal, run `vessl configure` to setup your environment.
3. A page on your browser will asking for access. Click **GRANT ACCESS**.
4. Choose your default organization.
5. Choose your default project.

### **Setup a** VESSL **environment**&#x20;

```
vessl configure [OPTIONS]
```

| Option                     | Description                                |
| -------------------------- | ------------------------------------------ |
| `-t`, `--access-token`     | Access token used for authorization        |
| `-o`, `--organization`     | Default organization                       |
| `-p`, `--project`          | Default project                            |
| `-f`, `--credentials-file` | Path to file containing configuration data |
| `--renew-token`            | Renew access token                         |

### **Change the default organization**

Set the default organization so that you do not enter the organization every time you execute the `vessl` command.

```
vessl configure organization [ORGANIZATION]
```

| Argument       | Description              |
| -------------- | ------------------------ |
| `ORGANIZATION` | New default organization |

### **Change the default project**

Set the default organization so that you do not enter the project every time you execute the `vessl` command.

```
vessl configure project [PROJECT]
```

| Argument  | Description         |
| --------- | ------------------- |
| `PROJECT` | New default project |

### **View the current configuration**

```
vessl configure list
```

### Configuration Precedence

You can configure your VESSL environment in multiple ways. Here is a list of ways, in order of precedence.

1. Command line arguments / options: `-o` , `-p` options and `ORGANIZATION` , `PROJECT` arguments.
2. Credentials file option: `-f` , a path to a file containing configuration data.
3. Environment variables: `VESSL_ACCESS_TOKEN`, `VESSL_DEFAULT_ORGANIZATION` .
4. Credentials file environment variable: `VESSL_CREDENTIALS_FILE` .
5. Default credentials file: a configuration file maintained by VESSL located at `~/.vessl/config` .

{% hint style="info" %}
Note that the `--renew-token` flag for `vessl configure` will take precedence over other methods and renew your access token.
{% endhint %}

