# SSH into UCL GPU machines

## The problem

UCL GPU machines (sentinel, warpath, etc.) are on an internal network. You can't reach them directly from the internet. You have to go through a *jump host* called `knuckles`:

```
Your laptop  →  knuckles.cs.ucl.ac.uk  →  sentinel.cs.ucl.ac.uk
                (public-facing)            (internal)
```

## Step 1: Generate an SSH key

On your laptop:

```bash
ssh-keygen -t ed25519 -C "your-email@ucl.ac.uk"
```

Press Enter to accept the default path (`~/.ssh/id_ed25519`). Set a passphrase or leave blank.

This creates two files:
- `~/.ssh/id_ed25519` — your private key (never share this)
- `~/.ssh/id_ed25519.pub` — your public key (this goes on the server)

## Step 2: Copy your public key to UCL

```bash
ssh-copy-id YOUR_USERNAME@knuckles.cs.ucl.ac.uk
```

You'll type your UCL password *this one time*. Because UCL machines share a networked home directory (NFS), this single command puts your key on knuckles, sentinel, and every other UCL CS machine simultaneously. That's why you never need a password again — every machine sees the same `~/.ssh/authorized_keys` file.

## Step 3: Create your SSH config

Create/edit `~/.ssh/config` on your laptop:

```
Host knuckles
    HostName knuckles.cs.ucl.ac.uk
    User YOUR_USERNAME
    IdentityFile ~/.ssh/id_ed25519

Host sentinel
    HostName sentinel.cs.ucl.ac.uk
    User YOUR_USERNAME
    ProxyJump knuckles
    RemoteCommand /bin/bash
    RequestTTY yes
    IdentityFile ~/.ssh/id_ed25519
```

What each line does:

| Directive | Purpose |
|-----------|---------|
| `HostName` | The actual server address |
| `User` | Your UCL username so you don't type it every time |
| `ProxyJump knuckles` | Automatically tunnels through knuckles to reach the internal machine |
| `RemoteCommand /bin/bash` | Forces bash because UCL's default shell is csh, which breaks things |
| `RequestTTY yes` | Required for `RemoteCommand` to work |
| `IdentityFile` | Points to your private key |

Add more hosts by copying the `sentinel` block and changing the `Host` and `HostName`.

## Step 4: Set permissions

SSH is strict about file permissions — it will refuse to use your key if the permissions are too open:

```bash
chmod 700 ~/.ssh
chmod 600 ~/.ssh/config
chmod 600 ~/.ssh/id_ed25519
```

## Step 5: Test

```bash
ssh sentinel
```

That's it. No password, no manually hopping through knuckles. SSH reads the config, connects to knuckles first, tunnels through to sentinel, authenticates with your key on both hops, and drops you into a bash shell.

## Step 6: VS Code Remote-SSH

1. Install the *Remote - SSH* extension in VS Code
2. `Cmd+Shift+P` (or `Ctrl+Shift+P`) > *Remote-SSH: Connect to Host*
3. Select `sentinel` from the list — VS Code reads your `~/.ssh/config` automatically

## Why no password is needed (summary)

Three things working together:

1. *SSH key pair* — your private key on your laptop proves your identity instead of a password
2. *Shared NFS home directory* — one `ssh-copy-id` to knuckles puts the key on all UCL machines
3. *macOS SSH agent* — caches your key passphrase in Keychain so you don't even type that after the first time (on Linux, run `ssh-add ~/.ssh/id_ed25519` to do the same)
