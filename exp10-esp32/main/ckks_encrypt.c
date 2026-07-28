#include "ckks_encrypt.h"

#include <stdlib.h>
#include <string.h>

esp_err_t ckks_encrypt_frame(const int16_t *pcm, size_t sample_count,
                              uint8_t **out_payload, size_t *out_len)
{
    size_t len = sample_count * sizeof(int16_t);
    uint8_t *buf = malloc(len);
    if (!buf) {
        return ESP_ERR_NO_MEM;
    }
    memcpy(buf, pcm, len);
    *out_payload = buf;
    *out_len = len;
    return ESP_OK;
}