# Client contract v1.1: `kv_transfer_params.glm53_spill`

A client tells the connector, per request, how long the conversation's resume points should be kept on NVMe and
which conversation and compaction generation the request belongs to. The server side of this contract is
`parse_spill_params` and the TTL/epoch logic in `code/glm53_nvme_spill_connector.py` (design: `docs/design.md`,
section 2.7). The field is optional; a request without it behaves exactly as before.

## The field

At the top level of the request body (the same for `/v1/responses` and `/v1/chat/completions`):

```json
"kv_transfer_params": {
  "glm53_spill": {
    "ttl_s": 86400,
    "session": "<the client's session id, an opaque string of at most 64 characters>",
    "epoch": 3
  }
}
```

| Key | Type | Meaning |
|---|---|---|
| `ttl_s` | integer >= 0 | Retention time in seconds. **0 = do not keep** (nothing is written, and existing resume points of the conversation are treated as expired). Counted from the last use (every use extends it) |
| `session` | string | Groups the requests of one conversation. The server uses it to clean up older resume points of the same session |
| `epoch` | integer >= 0 | +1 at every automatic compaction of the client. When the server sees a newer epoch it may drop resume points of older epochs of the same session (compaction changes the prefix, so they can no longer be hit) |

Values offered by the reference client:

| Choice | `ttl_s` |
|---|---|
| off (do not keep) | 0 |
| 1 hour | 3600 |
| 1 day | 86400 |
| 3 days | 259200 |

A request without the field gets the server defaults, i.e. the behaviour before the contract existed:
`SAVE_POLICY=repeat`, `MIN_TOKENS=65536`, no TTL, capacity LRU.

Never put secrets in the field. `session` is only the client's session id: no message text, keys or paths.

## Server-side validation

`parse_spill_params` is defensive and validates **each field on its own**. A malformed field is ignored and that field
takes the server default; the other fields are still used. It never raises and never fails a request.

| Field | Accepted | Ignored (field falls back to the default) |
|---|---|---|
| `ttl_s` | a real `int` with 0 <= `ttl_s` <= 604,800 (7 days) | `bool`, `float`, string, negative, **> 7 days** |
| `session` | a `str` of 1-64 characters | empty string, more than 64 characters, any other type |
| `epoch` | a real `int` >= 0 (the code also caps it below 2^53) | `bool`, `float`, string, negative |

If `kv_transfer_params` is not an object, `glm53_spill` is missing or not an object, or all three fields are
invalid, the request is treated as having no field at all.

What the server does with valid values (`docs/design.md`, section 2.7):

* `ttl_s > 0`: the resume point gets this TTL (the request's value replaces the stored one) and its `last_used` is
  refreshed by the first lookup of a request that extends it, by a restore and by a repeated save. Expired resume points
  are not used for restores and are deleted by a sweep every 30 s. With an explicit `ttl_s > 0` the first turn of a
  conversation is also written (`TTL_SAVE_POLICY=all`).
* `ttl_s = 0`: the request's resume point is not written, the request is not restored, and the resume points on its
  hash chain (same session or no session) and every resume point of the same session are expired.
* `epoch`: recorded per session at admission and at save time (not while a request only waits in the queue). Only a
  **strictly larger** epoch marks the session's older-epoch resume points as stale; stale ones are evicted first at the
  capacity cap and deleted after 30 minutes without use. A late request of the old epoch can still use them during
  that grace period.
* Older turns are pruned **per hash chain** (`KEEP_PER_CHAIN`), never "per session".

## How the field travels through vLLM

Both endpoints use vLLM's standard path. Line numbers are those of the production vLLM tree (the image's vLLM after the
launcher's start-up patches):

| Endpoint | Request field | Copied into sampling params |
|---|---|---|
| `/v1/responses` | `entrypoints/openai/responses/protocol.py:293` (`ResponsesRequest.kv_transfer_params`) | `to_sampling_params`, :429-431 -> `extra_args["kv_transfer_params"]` |
| `/v1/chat/completions` | `entrypoints/openai/chat_completion/protocol.py:475` | :701-703 -> `extra_args["kv_transfer_params"]` |

From there it reaches `Request.kv_transfer_params` (`v1/request.py:118`) and the connector.

* Correction to the first draft of the contract: `chat_completion/protocol.py:143` is the field of the **response**
  (`ChatCompletionResponse`), not of the request. The request side is :475, handed on at :701-703.
* `kv_transfer_params` is a declared field of both request models, so vLLM logs no "unknown field" warning.
* The field does **not** enter the prefix-cache hash: `v1/core/kv_cache_utils.py` never reads `extra_args`. Sending
  or not sending the field, or changing its values, does not split the prefix cache.
* If the server runs **without** a KV connector configured, vLLM logs `Got kv_transfer_params, but no KVConnector
  found` as a warning for every such request (`v1/engine/core.py:470`). The request does not fail. A client should
  stop sending the field in that case to keep the server log clean.
* The connector's `request_finished` returns `(False, None)`, so nothing is added to the response.

## Rules a client must follow

1. **Send the field on every request of a conversation**, including the client's own housekeeping requests (title
   generation, compaction summaries). Those requests carry the same `session` and `epoch` as the conversation.
   Consequence for any server implementation: it must **not** keep "only the newest resume point per session". A short
   title request would otherwise push out the conversation's resume point. Our connector prunes per hash chain
   (prefix relation) and drops older epochs only when it sees a strictly larger one.
2. **`epoch` +1 per successful compaction summary.** A failed compaction does not count. The summary request itself
   is sent with the **old** epoch (it reads the old context and may use the old resume point); the requests after it use
   the new epoch.
3. **Child sessions** (delegated sub-tasks that the client runs as separate conversations) **inherit the parent's
   `ttl_s` but send their own `session` and their own `epoch`.** Reason: a child's prompt is a different KV chain. If a
   child sent the parent's session id, the server would read the child's `epoch` 0 as an "older epoch" of the parent
   and could drop the parent's resume points.
4. `session` must be at most 64 characters. Longer or non-ASCII ids should be hashed by the client (the reference
   client sends `"sha256-"` followed by a digest truncated so that the whole string is 64 characters).
5. Changing the retention choice should not change the prompt. In the reference client the command's result is not part
   of the model input, so switching it does not break the prefix cache.

## Reference client: DSH

DSH (DeepSeek Harness, github.com/deepseek-ai/deepseek-harness) is the client we built the contract with. The DSH
patches themselves are **not included in this repository**, because they are against a locally modified DSH build and
would not apply to upstream DSH. This section describes how the change was built so that it can be reproduced in any
client.

DSH talks to the server through `/v1/responses` (its provider uses the OpenAI Responses API), not through
`/v1/chat/completions`. Both paths carry the field identically (table above).

### What was changed

* **Per-session menu.** The model/effort menu got one more row, "NVMe retention", with a sub-screen offering off /
  1 hour / 1 day / 3 days. A session where nothing was chosen uses the default from the settings (initially 1 hour).
  The choice is stored as a command entry in the session log, so it survives a page reload and a client restart. The
  switch appears in the conversation as one line, like other mode switches.
* **`/nvme` command.** `/nvme 1d`, `/nvme default` and so on can be typed instead of using the menu. The command
  result is not sent to the model (measured: it does not enter the request input), so switching does not break the
  prefix cache. The command is registered only while at least one provider has the feature enabled, and follows
  settings hot reload. Child sessions cannot use it (it returns an error).
* **Where the field is added.** There is a single insertion point right before sending: the provider library's
  `onPayload` hook appends `kv_transfer_params` as the **last key** of the request body. Every other field stays
  byte-for-byte the same. If resolving the values fails, the request is still sent, just without the field.
* **Settings switch.** `llm-pi-ai.providers.<name>.nvmeSpill: {enabled, defaultTtlS}`. The feature is active only for a
  provider where `enabled: true` is written explicitly (the settings schema library fills an omitted object with `{}`,
  which was found by measurement, so "present" could not be used as the switch). `enabled: false` or removing the
  section stops sending the field without a restart (settings hot reload, measured); the menu row disappears as well.
  Changing `defaultTtlS` applies to sessions that never chose a value, again without a restart.
* **Values.**
  * `ttl_s`: the session's choice, otherwise `defaultTtlS`.
  * `session`: the DSH session id as is. If it is longer than 64 characters or not ASCII, it is replaced by
    `"sha256-"` + a 57-character digest (64 characters in total). DSH ids are 36-44 characters, so in practice they pass
    through unchanged.
  * `epoch`: counted from the session's own log: +1 for each confirmed compaction summary; failed compactions do not
    count; a prune-type compaction (disabled in the default preset) counts +1 per prune. The summary request uses the
    old epoch and the next request the new one.
  * Child sessions inherit `ttl_s` by walking the parent-session chain through live sessions, up to 16 levels; if the
    parent is not loaded, the default applies. `session` and `epoch` are the child's own.
* **Robustness.** Strings taken from the session log are cut to 160 characters before any regular expression runs
  (anchored patterns, bounded quantifiers); a 2 MB pathological input is processed in under 50 ms (unit test).

### Measured request bodies

Captured against a mock server with the final code (end-to-end run on a fresh home directory). Key order of the body:
`model, input, stream, prompt_cache_key, store, max_output_tokens, tools, reasoning, include, kv_transfer_params`
(only the last key is new).

| Operation | `glm53_spill` |
|---|---|
| New session, nothing chosen (title generation too) | `{"ttl_s":3600,"session":"<session id>","epoch":0}` |
| Choose 1 day / off / 3 days / 1 hour | `ttl_s` 86400 / 0 / 259200 / 3600 |
| Choose 1 day, reload the page, then restart the client | 86400 / 86400 (the menu also shows "1 day") |
| `/compact` (manual): summary request -> next request | epoch 0 -> epoch 1 |
| Automatic compaction (test preset with a very low threshold), confirmed twice | epoch 0 -> 1 -> 2 (monotonic; a failed compaction does not increase it) |
| Two kinds of child session | `{"ttl_s":86400,"session":"<child session id>","epoch":0}` for each, with its own id |
| Session with nothing chosen, `defaultTtlS` changed to 259200 (hot reload) | 259200 (a session that chose 1 day keeps 86400) |
| `enabled: false` / section removed (hot reload) | No field, the menu row disappears. Restoring the setting brings both back |
| `/nvme 1d`, then `/nvme default` | 86400 -> 3600 |

Compared with the unmodified build (same text, same settings, session ids / timestamps / install paths normalized):
for conversation and title requests the **only** difference is the appended `kv_transfer_params`; headers are
identical. With the feature not configured, the modified build sends a body **byte-identical** to the unmodified
build.

### Tests

* Unit tests: 12/12 (projection of the session log, epoch counting, insertion, inheritance including cycles, the
  command, the settings schema, folding of a real session log, pathological input).
* End-to-end run (fresh home, mock server only): all checks pass (every row of the table above plus three body
  comparisons).
* Rollback compatibility: a session that used `/nvme` in the modified build opens and sends normally in the
  unmodified build (no field, no errors or warnings); switching back restores "1 day, epoch 1". The unmodified build
  starts and sends normally with the `nvmeSpill` section present in its settings.
* The client's existing test suites gave the same sets of passing and failing tests with and without the change
  (no regression).

## Minimal client example

Language-neutral in spirit; written as Python.

```python
import hashlib

CHOICES = {"off": 0, "1h": 3600, "1d": 86400, "3d": 259200}

def spill_session_id(raw_id: str) -> str:
    if len(raw_id) <= 64 and raw_id.isascii():
        return raw_id
    return "sha256-" + hashlib.sha256(raw_id.encode()).hexdigest()[:57]

def spill_field(conv, default_ttl_s=3600):
    """conv: the conversation the request belongs to (also for title/summary requests)."""
    src = conv
    while src.choice is None and src.parent is not None:
        src = src.parent          # child sessions inherit only the TTL
    ttl_s = CHOICES.get(src.choice, default_ttl_s)
    # epoch = successful compaction summaries of THIS conversation; a summary request
    # is built before the counter is incremented, so it carries the old epoch
    return {"glm53_spill": {"ttl_s": ttl_s,
                            "session": spill_session_id(conv.id),
                            "epoch": conv.completed_summaries}}

def build_body(conv, body: dict) -> dict:
    body = dict(body)
    body["kv_transfer_params"] = spill_field(conv)   # append as the last key
    return body
```
