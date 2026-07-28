#pragma once

#include <stddef.h>
#include <stdint.h>
#include "esp_err.h"

/*
 * PLACEHOLDER (phase 1): passes PCM through unencrypted. The real CKKS
 * encoder/encryptor -- interoperable with the OpenFHE parameters/public key
 * used server-side elsewhere in this repo -- is follow-up work. See
 * exp10-esp32/README.md. Until that lands, nothing sent by this firmware
 * is confidential.
 *
 * Caller owns *out_payload and must free() it.
 */
esp_err_t ckks_encrypt_frame(const int16_t *pcm, size_t sample_count,
                              uint8_t **out_payload, size_t *out_len);