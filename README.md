# Backdrop replacement service

Given two samples of the **same animation frame** over two solid colours, this
service recovers which foreground/alpha combinations can explain both samples
and composites the frame over a new background — but only where the new RGB is
provably unique. Everywhere else it returns a mask that distinguishes

* **contradiction** — no foreground/alpha pair can reproduce both samples
  (the samples were mis-registered or the solid colour was entered wrongly);
* **ambiguous** — the samples are consistent, but more than one foreground can
  produce them, so one or more channels of the new RGB are not determined.

## Quick start (Docker Compose)

```sh
docker compose up --build
```

The API listens on <http://localhost:8000> (interactive docs at `/docs`).

Then run the bundled end-to-end request (needs `curl`):

```sh
python3 examples/make_demo.py     # regenerate the sample PNGs
BASE=http://localhost:8000 examples/demo/request.sh
```

Both scenes are committed under `examples/demo/`, so only `request.sh` is
needed if the PNGs are already present.

`request.sh` registers two samples per scene and posts a new background. The
`cel` scene returns an image; the `soft` scene returns a mask.

## API

All image uploads must be **8-bit RGB PNG** (colour type 2, bit depth 8),
at most 1024 pixels per side. Errors use HTTP 4xx with
`{"detail": {"code": ..., "message": ...}}`.

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/healthz` | liveness |
| `GET` | `/v1/samples` | both slots, backgrounds, versions, `ready` flag |
| `PUT` | `/v1/samples/{slot}` | register/correct slot `1` or `2`; multipart `image`, `background` |
| `DELETE` | `/v1/samples/{slot}` | drop a slot |
| `POST` | `/v1/replace` | multipart `background` PNG, optional `format=json\|png` |

`background` accepts `#RRGGBB`, `RRGGBB`, `r,g,b` or `rgb(r,g,b)`.

### Correcting a sample

`PUT` on an existing slot replaces it atomically. If a replacement computation
is running when a sample changes, the request finishes with
`409 {"detail": {"code": "samples_changed", ...}}` and the client should
simply rerun `/v1/replace`; the new sample is kept and no stale image is
returned.

### Response

`format=json` returns the result as base64 PNGs:

```json
{
  "status": "ok" | "mask",
  "reason": null | "not_unique",
  "width": 192, "height": 192,
  "image_png_base64": "...",         // only when status = ok
  "mask_png_base64": "...",          // only when status = mask (RGBA preview)
  "mask_regions_png_base64": "...",  // only when status = mask (lossless)
  "mask_regions": {"red_channel": "contradiction",
                   "green_channel": "ambiguous",
                   "blue_channel": "unique"},
  "counts": {"unique": 0, "contradiction": 0, "ambiguous": 0, "total": 0},
  "versions": [1, 2]
}
```

The preview mask is transparent where the result is unique, opaque red
`#ff4040` for contradictions and opaque blue `#4080ff` for ambiguous pixels.
`mask_regions_png_base64` is an exact diagnostic: red = contradiction,
green = ambiguous, blue = unique.

With `format=png` the response body is the composed image (`X-Compose-Status:
ok`), or the preview mask (`X-Compose-Status: mask`) with
`X-Unique-Pixels`, `X-Contradiction-Pixels`, `X-Ambiguous-Pixels` headers.

### One request with curl

```sh
curl -X PUT localhost:8000/v1/samples/1 \
  -F 'background=#132a5e' -F 'image=@examples/demo/soft/sample_1.png'
curl -X PUT localhost:8000/v1/samples/2 \
  -F 'background=#e8c16b' -F 'image=@examples/demo/soft/sample_2.png'
curl -X POST localhost:8000/v1/replace \
  -F 'background=@examples/demo/soft/background.png' -F 'format=png' \
  -o result.png
```

## How the solver works

For a pixel with shared 8-bit alpha `A`, foreground channel `F` and sample
colour `s` over background `b`, the 8-bit rounding rule

```
s = (A*F + (255-A)*b + 127) // 255
```

is exactly

```
(255*s - 127) - (255-A)*b  <=  A*F  <=  (255*s + 127) - (255-A)*b
```

Intersecting the windows of the two samples gives, for each channel and each
alpha, one integer interval of admissible `F`; the three channels are
independent at a fixed `A`, so a pixel is feasible for alpha `A` iff all three
intervals are non-empty. Over the new background each channel then ranges over
one output interval, and the pixel is **unique** iff those intervals collapse
to one value for every feasible alpha (all three channels). `A = 0` is handled
separately because the foreground is unobservable there.

The implementation is exact but does not expand 256 candidates per pixel:

1. Linear inequalities in `A` are intersected first, giving a candidate alpha
   interval per pixel (checked to be a superset of every feasible alpha).
2. Only alphas inside that interval are evaluated, one alpha at a time on a
   row chunk of about 128k pixels; per-alpha output bounds are accumulated
   as 8-bit minima/maxima.
3. A pixel whose output interval has already widened is dropped from further
   alpha scans, since it can no longer become unique.

Peak extra memory is a fixed set of planes per chunk (tens of MB at most,
measured ~36 MB for 1024×1024), independent of the 256 alpha candidates.
A 1024×1024 image with an entirely zero region (the worst case, where all 255
alphas must be examined) runs in roughly 5–6 s on the development machine.

## Limitations

* Both samples must have the same size. Registering a differently sized sample
  is allowed (so it can be corrected next) but is flagged with
  `pair_size_match: false` and `/v1/replace` answers `409 sample_size_mismatch`
  until the pair matches. A new background of the wrong size is rejected the
  same way.
* A mistaken solid-colour label usually turns most pixels into contradictions
  rather than silently producing a wrong image, but a wrong label that still
  fits some alphas can make results ambiguous — both are flagged.
* State is in memory only (one process); restarting the container clears the
  registered samples.

## Development

```sh
python3 -m pytest tests/ -q
uvicorn app.api:app --reload
```
