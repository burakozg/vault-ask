# Setting up the embedding host

vault-ask's embedding model runs on a machine on your LAN, not in the NAS
container — see README "Models". This is that machine's setup, once. Any
always-on-ish Mac or Linux box on the same network as the NAS works; it does
not need to be the NAS itself, and it does not need a GPU (`bge-m3` is a small
model — a few hundred MB, runs fine on CPU).

vault-ask reads the address from `VAULTASK_MODELS__EMBEDDING_BASE_URL` — see
"Point vault-ask at it" below — so nothing here is hardcoded into the app; the
same instructions work whichever machine you pick, and moving it later is a
one-line config change.

**For local development**, if the machine you're coding on already has Ollama
running (check with `pgrep -fl ollama` or `curl http://127.0.0.1:11434/api/version`),
you can skip straight to step 2 (`ollama pull bge-m3`) and point
`VAULTASK_MODELS__EMBEDDING_BASE_URL` at `http://127.0.0.1:11434` — no LAN
exposure needed for a dev loop that never leaves your own machine. Steps 3–5
(binding to the LAN, finding the IP, verifying from another machine) are only
needed for the real deployment, where the NAS container is a different machine
from whatever runs Ollama.

## 1. Install Ollama

macOS: download from https://ollama.com/download, or `brew install ollama`.
Linux: `curl -fsSL https://ollama.com/install.sh | sh`.

## 2. Pull the embedding model

```sh
ollama pull bge-m3
```

`bge-m3` is multilingual (matters here — the vault carries Turkish and
Swedish) and produces 1024-dimensional vectors, which is what
`models.embedding_dim` in `config.yaml` assumes. If you ever change the model,
update `embedding_dim` to match and run `vault_ask index --rebuild` — see
README "Index" on why a changed embedding model forces a rebuild rather than
silently mixing vector spaces.

## 3. Make Ollama listen on the LAN, not just localhost

Ollama's default bind is `127.0.0.1:11434` — reachable from the machine
itself, not from the NAS container. It has to listen on an interface the rest
of the network can reach.

**macOS (the Ollama.app menu-bar app):** open Ollama's Settings and turn on
**Expose Ollama to the network**, then quit and reopen the app if it asks. The
setting is stored by the app and applied at every start, so it survives reboots
with nothing else to install. Verified: with the toggle on and `OLLAMA_HOST`
unset in launchd, a fresh process listens on `*:11434`, and `localhost`, the
LAN address and the NAS all reach it.

Do not use `launchctl setenv OLLAMA_HOST` for the app instead. It is forgotten
at reboot, and a LaunchAgent that sets it at login races the app: Ollama can
start first and come up on `127.0.0.1` only, with the variable correctly set and
no effect, so the NAS sees "connection refused" and vault-ask quietly falls back
to keyword-only search (and its indexer logs a failure for every doc).

**Linux, or macOS running `ollama serve` yourself (e.g. as a service):**

```sh
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

`0.0.0.0` binds every interface on the box, loopback included, which keeps
`localhost:11434` working for everything else on the machine (other projects
here default to it). If the machine has more than one non-loopback interface
(e.g. Wi-Fi and Ethernet both up) and you want to restrict it, bind a specific
LAN IP instead: `OLLAMA_HOST=10.0.0.x:11434`.

**Binding a LAN IP has two costs.** It stops listening on loopback, so
anything pointed at `localhost:11434` or `127.0.0.1:11434` (a local `.env`, the
`ollama` provider in security-digest) gets connection refused until it is
repointed at the LAN address. And DHCP can hand the machine a new address,
after which Ollama silently fails to bind and the NAS sees "connection
refused" with nothing in vault-ask having changed. On a single-interface
machine, `0.0.0.0` exposes exactly the same network surface and has neither
problem.

**Ollama has no authentication.** Listening on the LAN means anything on the
LAN can run models and read the model list. The macOS application firewall
cannot restrict by source address; if that matters, limit port 11434 to the
NAS's address with a `pf` rule or at the router.

macOS's firewall may prompt to allow incoming connections for `ollama` the
first time something on the LAN reaches it — allow it, or the request will
hang rather than fail cleanly.

## 4. Find the machine's LAN IP

```sh
# macOS, Wi-Fi:
ipconfig getifaddr en0
# macOS, Ethernet (varies by machine — en1, en5... ifconfig -a to list):
ipconfig getifaddr en1
# Linux:
ip -4 addr show | grep inet
```

This IP can change on DHCP renewal. If it does, update
`VAULTASK_MODELS__EMBEDDING_BASE_URL` (step 6) — nothing else needs to change.
A static DHCP reservation on your router avoids this entirely and is worth
doing once you're happy with which machine hosts this.

## 5. Verify it's reachable from another machine

From your Mac (or wherever you're deploying vault-ask from), not from the
Ollama host itself — the point is to prove the network path, not that the
process is running:

```sh
curl http://<the-ip-from-step-4>:11434/api/embeddings \
  -d '{"model": "bge-m3", "prompt": "test"}'
```

A JSON body with an `"embedding"` array of 1024 floats means it works. A
connection refused or timeout means step 3 didn't take — check `OLLAMA_HOST`
is actually set in the environment the Ollama process is running in
(`launchctl getenv OLLAMA_HOST` on macOS), and that nothing on the host
firewall is blocking port 11434 from the LAN.

## 6. Point vault-ask at it

In vault-ask's `.env` (see `.env.example` — never commit this file):

```sh
VAULTASK_MODELS__EMBEDDING_BASE_URL=http://<the-ip-from-step-4>:11434
```

`vault_ask index` will use this the moment step 3 (embeddings) lands — see
README "Build order". Until then this variable is read but unused.

## Things that will bite you later

- **The machine sleeping.** Indexing runs hourly (README "Deployment"); if the
  embedding host is asleep when it fires, that run's embedding calls fail and
  the run is skipped — no data is lost, it just retries next hour (same
  failure shape as podcast-digest's remote ASR/TTS hosts, which this design is
  copied from). On a laptop, disable sleep-on-lid-closed while plugged in, or
  pick a machine that's normally on.
- **Ollama auto-updating and changing its bind behaviour.** Re-run step 5
  after any Ollama update if indexing starts failing.
- **This is a LAN-only address, deliberately.** `personal` chunks reach this
  machine during indexing regardless of `allow_web` — that's a property of
  embedding running locally, not a new exposure it creates — but it does mean
  the embedding host itself should not be more widely reachable than "the home
  network," the same trust boundary the NAS itself sits inside.
