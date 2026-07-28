#pragma once

#include <stddef.h>
#include <stdint.h>
#include "esp_err.h"

/* Configures the I2S peripheral for a mono, 16-bit digital mic (INMP441 etc). */
esp_err_t audio_capture_init(void);

/* Blocks until up to max_samples int16 PCM samples are available. */
esp_err_t audio_capture_read(int16_t *out, size_t max_samples, size_t *out_count);

void audio_capture_deinit(void);
