# Security and privacy

OneLoad accepts a JSON manifest of at most 256 KiB containing up to 32 narration
scenes. It rejects unknown fields, caps text and generation settings, and limits
one manifest to 16,384 requested tokens. Scene identifiers use a restricted
character set. Output paths must be relative WAV names whose components use the
portable ASCII set `A-Z`, `a-z`, `0-9`, `.`, `_`, and `-`; case-folded duplicates
are rejected. Manifest and caption inputs must be regular files. Every path
component is opened descriptor-relatively with `O_NOFOLLOW`, and the final file
is read through a bounded nonblocking descriptor, so links, FIFOs, and devices
cannot redirect or indefinitely stall the read. Audio duration is capped and
written through a private file descriptor. OneLoad unlinks the random staging
name before writing, then publishes directly from that descriptor into a
previously absent destination with macOS `fclonefileat`. Existing destinations
are never overwritten, and the committed descriptor and bytes are verified.

Before downloading, OneLoad creates or opens the target without following links,
requires an owner-protected directory chain, and rejects links, special files,
unexpected entries, and unprotected model files. OneLoad's downloader therefore
receives only a validated target. It pins the public Hub endpoint,
disables implicit authentication, and removes inherited token and staging
variables, so this public download cannot forward an ambient Hugging Face token.
Only paths named by the lock are requested. Every response must use HTTPS, stay
within its locked byte budget, and match the locked SHA-256 before a private
descriptor is published; the exact finished snapshot is validated again before
the downloader reports success. Download metadata caches are not required.
The model lock records an immutable model revision plus exact byte sizes and
SHA-256 digests for the complete downloaded snapshot outside Hugging Face's
private local cache. OneLoad traverses only directory prefixes named by that
lock, skips the root `.cache` without descending, and rejects the first missing,
extra, linked, or changed entry. It binds the model root and each source
directory by descriptor, then opens every leaf with `O_NOFOLLOW` and
`O_NONBLOCK`. Hashing and fallback copying consume exactly the locked byte count
and reject early EOF or growth. It freezes every required file into a private
same-filesystem copy-on-write clone
(or a private copy when cloning is unavailable), verifies the frozen bytes, and
gives only that bound view to MLX. It checks those bound identities after loading
and again after generation. The loader view and output staging require an
owner-bound parent that is not group/world writable, or a trusted sticky parent;
macOS extended allow ACLs are rejected, and create/open identities are compared
before use. Local operational receipts expose the public model identity, revision,
license, relative output names, timing, and hashes, but never the local model
directory. The publishable benchmark substitutes ordinal scene labels. Its report and stable failure messages similarly exclude usernames,
device identifiers, terminal controls, absolute paths, network URLs, and private
child-process commands.

Public shell entry points must be invoked by their documented relative path from
the repository root. They reject absolute paths, aliases, links, and hard links.
Both start the fixed `/bin/bash` interpreter in privileged mode, which ignores
inherited shell startup files and functions, and invoke the project Python directly
instead of resolving a tool through `PATH`. An isolated system Python check opens the current directory and every
repository-relative script component with `O_NOFOLLOW`, then compares the final
regular file to the entry point already opened by Bash. WAV and benchmark JSON
writes walk and create every output-root
component from `/` or the already-open current directory with `O_NOFOLLOW`;
both unlink exclusive random mode-0600 staging names before use and publish
through descriptor-bound, no-overwrite copy-on-write clones. Output volumes must
support that macOS cloning primitive. Benchmark
children run under an ACL-checked private temporary root and receive a mode-0400
copy of the exact manifest bytes validated by the parent. They use isolated
Python import semantics, ignore the working directory and user site, and receive
no inherited `BASH_ENV`, `ENV`, `PYTHONHOME`, `PYTHONPATH`, or `PYTHONSTARTUP`.
The benchmark admits at most 16 render subprocesses and applies one 30-minute
deadline across the complete run. Child stdout and stderr are byte-bounded, each
child runs in a separate process group, and every receipt must match a strict
schema and the same manifest SHA-256. The parent independently inventories and
hashes the expected private WAV files. Only parent-measured end-to-end wall time
is used as performance evidence; child-reported timing and memory fields are
validated but discarded. The final report also binds the model-lock and
dependency-lock SHA-256 values, verified model byte count, and hashes of the
runtime implementation. Its chip probe uses the fixed `/usr/sbin/sysctl` path
with a five-second timeout.

After the model is downloaded, the documented benchmark runs with Hugging Face
offline mode and telemetry disabled. It makes no paid API call and sends no
narration text to an external service. Model weights, virtual environments,
generated WAV files, caches, and temporary benchmark directories are excluded
from version control.
