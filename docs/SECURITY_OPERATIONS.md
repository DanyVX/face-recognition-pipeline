# Security operations

How to run the service with its privacy controls on, and what each control does and does not do.
See `THREAT_MODEL.md` for the reasoning and `DESIGN.md` D-028..D-031 for the decisions.

## 1. Encryption keys

Generate a key (it is printed, not stored anywhere):

```bash
uv run frp keys generate-encryption-key --id k1
```

Provide it to the service through the environment (preferred) or a key file:

```text
ENCRYPTION_KEYS=k1:BASE64KEY            # id:key pairs, comma separated
ENCRYPTION_ACTIVE_KEY=k1                # which one seals NEW data (defaults to the first)
# or a JSON key file: {"active": "k1", "keys": {"k1": "BASE64KEY"}}
ENCRYPTION_KEY_FILE=/run/secrets/frp_keys.json
```

`ENCRYPTION_ACTIVE_KEY` defaults to the first key listed in `ENCRYPTION_KEYS`; with a key file the
file's own `active` field decides.

- The service **refuses to start** with no key. `ALLOW_UNENCRYPTED=true` is a development opt-out
  and logs a warning.
- `ALLOW_KEY_GENERATION=true` lets a *missing* key file be created on first start. It exists for
  the demo container; do not use it in production, because a key generated next to the data it
  protects protects nothing against disk theft.
- Keep keys **off the data volume** and out of git. Lose the key and the embeddings are
  unrecoverable (identities can still be listed and erased).

## 2. Rotating a key

1. Generate a new key: `uv run frp keys generate-encryption-key --id k2`.
2. Set `ENCRYPTION_KEYS=k2:NEW,k1:OLD` and `ENCRYPTION_ACTIVE_KEY=k2` (old key stays listed so old
   blobs can still be opened).
3. Stop the HTTP service first: the command is a separate process, and a running service keeps its
   own in-memory index and would not see the change.
4. Run `uv run frp encryption migrate`. It re-seals every embedding and every **registered**
   snapshot with `k2` and prints counts, including `key_usage_after`.
5. When `key_usage_after` shows only `k2`, remove `k1` from `ENCRYPTION_KEYS`.

The same command encrypts a store that was created with `ALLOW_UNENCRYPTED=true`. Before you
migrate, back up the data directory; the operation rewrites rows in place.

Rotation does **not** touch old backups: a backup taken before rotation is still readable with the
old key. Destroy old backups or the old key according to your retention rule.

## 3. Erasure ("delete this person")

```bash
uv run frp erase --identity <id>            # service stopped
# or, on the running service:
curl -X DELETE -H "Authorization: Bearer $KEY" http://localhost:8000/v1/identities/<id>
```

What it does, in order, and verifies afterwards (response `erasure_verified`; the CLI prints a full
report): zero + delete the database rows and vacuum, rebuild the in-memory index, delete sealed raw
images, rewrite registered index snapshots without the person, purge idempotency (replay) records,
redact the identity id in earlier audit events, and write an `erase` audit event. Any check that
fails raises `ERASURE_NOT_VERIFIED` rather than reporting success.

What it **cannot** reach:

- Backups and copies of the data directory (including cloud snapshots of the volume).
- Index snapshots that were not registered. Create them with `uv run frp snapshot PATH` (which
  registers them); a file written by other means is invisible to erasure.
- Remnants outside the application's control: filesystem journals, SSD wear-levelling, OS caches,
  swap. `secure_delete` and vacuum remove the data from the SQLite file, not necessarily from the
  physical medium. Full-disk encryption is the control for that.
- Data an API client copied (the service returns embeddings only when explicitly configured to).

Erasure runs through the service, which needs the key to start (the gallery decrypts everything at
startup). If the key is lost the data is unreadable anyway; remove the data directory instead.

## 4. Audit log

`audit.db` in the data directory records time, request id, API-key id, action, identity id and
outcome - never images, embeddings or names. Read it with `uv run frp audit [--identity ID]`.
It is **not** tamper-evident: protect the file with OS permissions and ship it elsewhere if you
need integrity. `AUDIT_ENABLED=false` turns it off.

## 5. Raw images

Off by default (`STORE_RAW_IMAGES=false`). If enabled they are stored sealed with the same keys,
under `data/raw/`, and are removed by erasure. Enable only with a documented reason and retention.

## 6. Container hardening

The Dockerfile runs as a non-root user and the compose file uses a read-only root filesystem with
writable volumes only for `/data` and a tmpfs for `/tmp`. These files have been **written but not
built or run on this development machine** (no Docker available); the CI `docker` job is the
first place they are exercised. Check `docs/LIMITATIONS.md` for current status.

## 7. Checklist before pointing real faces at it

- [ ] You have the right to process these people's faces and they consented.
- [ ] Keys are outside the data volume; `ALLOW_UNENCRYPTED` is unset.
- [ ] TLS terminates in front of the service.
- [ ] `RETURN_EMBEDDINGS` and `STORE_RAW_IMAGES` are off unless needed.
- [ ] Thresholds were chosen on *your* data (the shipped ones are provisional).
- [ ] Backups are encrypted and covered by your erasure procedure.
- [ ] You are not using the InsightFace pretrained models commercially (non-commercial research
      use only).
