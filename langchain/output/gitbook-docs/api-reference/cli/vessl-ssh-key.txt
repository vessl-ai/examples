# vessl ssh-key

### Overview

Run `vessl ssh-key --help` to view the list of commands, or `vessl ssh-key [COMMAND] --help` to view individual command instructions.

### Add a SSH public key

```
vessl ssh-key add [OPTIONS]
```

| Option | Description         |
| ------ | ------------------- |
| --path | SSH public key path |
| --name | SSH public key name |

```bash
$ vessl ssh-key add
[?] SSH public key path: /Users/johndoe/.ssh/id_ed25519.pub
[?] SSH public key name: john@abcd.com

Successfully added.
```

### List ssh public keys

```bash
vessl ssh-key list
```

```bash
$ vessl ssh-key list
 NAME           FINGERPRINT                   CREATED
 john@abcd.com  SHA256:ugLx91zLE9ELAqT19uNjQ  6 hours ago
```

### Delete a ssh public key

```bash
vessl ssh-key delete
```

```bash
$ vessl ssh-key delete
[?] Select ssh public key: john@abcd.com / SHA256:ugLx91zLE9ELAqT19uNjQ (created 6 hours ago)
 > john@abcd.com / SHA256:ugLx91zLE9ELAqT19uNjQ (created 6 hours ago)
 
 Successfully deleted.
```
