# SSH Connection

Once you run a workspace, you can fully leverage the development environment using SSH.

![](<../../.gitbook/assets/image (75).png>)

![](<../../.gitbook/assets/image (198).png>)

### 1. Create SSH Key

To enable SSH connection, you first need a SSH key pair. Once you obtained a public key for your account and workspace instance, you can connect it with a private key.&#x20;

```
$ ssh-keygen -t ed25519 -C "vessl-floyd"
Generating public/private ed25519 key pair.
Enter file in which to save the key (/Users/floyd/.ssh/id_ed25519):
Enter passphrase (empty for no passphrase): 
Enter same passphrase again: 
Your identification has been saved in /Users/floyd/.ssh/id_ed25519.
Your public key has been saved in /Users/floyd/.ssh/id_ed25519.pub.
The key fingerprint is:
SHA256:78yjMGcJoV73v/jkLHIRhdC0wL0FBL6c68T0MZGoV2Q savvihub-floyd
The key's randomart image is:
+--[ED25519 256]--+
|       .+BEo     |
|       ..=+oo    |
|      . o =+     |
|     . + +o.     |
|    . + S o.     |
|   . . * *.o     |
|    . o B +..    |
|       B.++*     |
|        o+=+*.   |
+----[SHA256]-----+
```

### 2. Add SSH public key to your VESSL account

You can add your SSH public key to your account using VESSL CLI. The added keys will be injected to every running workspaces you created. You can manage your keys with `vessl ssh-keys list` and `vessl ssh-keys delete` commands.

```
$ vessl ssh-keys add      
[?] SSH public key path: /Users/floyd/.ssh/id_ed25519.pub
[?] SSH public key name: vessl-floyd

Successfully added.
```

### 3. Connect via CLI

If there are more than one running workspaces, you will be asked to select one to connect.

```
$ vessl workspace ssh
The authenticity of host '[tcp.apne2-prod1-cluster.savvihub.com]:30787 ([52.78.240.117]:30787)' can't be established.
ECDSA key fingerprint is SHA256:iSexO7W1U14P3Pp6wRfPleHABQQMek/JAgb5kHqg5Jw.
Are you sure you want to continue connecting (yes/no/[fingerprint])? yes
Welcome to Ubuntu 18.04.5 LTS (GNU/Linux 4.14.238-182.422.amzn2.x86_64 x86_64)

 * Documentation:  https://help.ubuntu.com
 * Management:     https://landscape.canonical.com
 * Support:        https://ubuntu.com/advantage
This system has been minimized by removing packages and content that are
not required on a system that users do not log into.

To restore this content, you can run the 'unminimize' command.

The programs included with the Ubuntu system are free software;
the exact distribution terms for each program are described in the
individual files in /usr/share/doc/*/copyright.

Ubuntu comes with ABSOLUTELY NO WARRANTY, to the extent permitted by
applicable law.

vessl@workspace-9670nnkn5l16-0:~$ 
```

### 4. Setup VSCode Remote-SSH plugin config

You can also add your workspace to VSCode Remote-SSH plugin config. `vessl workspace vscode` adds the information to `~/.ssh/config` so that the workspace can show up in the host list.

```
$ vessl workspace vscode
Successfully updated /Users/floyd/.ssh/config

$ cat ~/.ssh/config
Host acceptable-bite-1627438220
    User vessl
    Hostname tcp.apne2-prod1-cluster.savvihub.com
    Port 30787
    StrictHostKeyChecking accept-new
    CheckHostIP no
    IdentityFile /Users/floyd/.ssh/id_ed25519
```

![](<../../.gitbook/assets/image (78).png>)

![](<../../.gitbook/assets/image (80).png>)

![](<../../.gitbook/assets/image (81).png>)

### 5. Manual Access&#x20;

You can integrate with other IDEs and make SSH connection without VESSL CLI using the host, username, and port information. In this case, the host is `tcp.apne2-prod1-cluster.savvihub.com`, username `vessl`, and port `30787`. The full SSH command is `ssh -p 30787 -i ~/.ssh/id_ed25519 vessl@tcp.apne2-prod1-cluster.savvihub.com`.
