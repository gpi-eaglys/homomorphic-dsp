# Experiment 10: ESP32 audio 

* this experiment builds a firmware for esp32 that 
* records audio from a I2C mic
* encrypts audio under CKKS using a bundled key
* streams the ciphertext to a server 


## Status: Phase 1 (this commit)

OpenFHE itself cannot run on an ESP32 (far too much RAM/flash, relies on
things the ESP-IDF toolchain doesn't provide), so a CKKS encoder/encryptor
needs to be ported to C for the device, interoperable with the OpenFHE
ring/parameters/public key used server-side. That's a substantial piece of
work (polynomial ring arithmetic, NTT, sampling) and hasn't been built yet.

This first pass gets the rest of the pipeline running end-to-end:

- I2S digital mic capture (`main/audio_capture.c`)
- WiFi station + TCP client to the server (`main/net_client.c`)
- `main/ckks_encrypt.c` is a **passthrough stub** — see the warning below.

**`ckks_encrypt_frame()` does not encrypt anything yet.** Raw PCM audio is
sent over a plain TCP socket. Do not point this at anything other than a
trusted local network/test server until the real CKKS module lands.

## Hardware

Wired for an I2S MEMS mic (INMP441 / SPH0645 or similar):

| Signal          | ESP32 GPIO (default, see Kconfig) |
|-----------------|------------------------------------|
| WS / LRCLK      | 15                                  |
| SCK / BCLK      | 14                                  |
| SD / DOUT (mic) | 32                                  |

Tie the mic's L/R select pin to GND (left slot) to match
`I2S_STD_SLOT_LEFT` in `audio_capture.c`. Adjust pins via `idf.py menuconfig`
under "Homomorphic DSP ESP32 (exp10) Configuration" if wired differently.

## Build

Requires the ESP-IDF toolchain (v5.x — uses the `driver/i2s_std.h` API).
Not installed in this dev environment, so this hasn't been build-tested yet;
verify once ESP-IDF is set up.

```
. $IDF_PATH/export.sh
cd exp10-esp32
idf.py set-target esp32
idf.py menuconfig   # set WiFi SSID/password and server host/port
idf.py build flash monitor
```

## TODO

- [ ] Pick CKKS parameters shared with the OpenFHE server config and
      generate/provision a public key onto the device (or fetch at boot).
- [ ] Port CKKS encode (canonical embedding) + encrypt to C for `ckks_encrypt.c`.
- [ ] Server-side TCP listener that decodes frames and feeds the FHE pipeline
      (doesn't exist yet — nothing in this repo currently listens for this).
- [ ] Framing/protocol between device and server (currently a raw byte stream,
      no length prefixing or session handshake).