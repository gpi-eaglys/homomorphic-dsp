#include <stdlib.h>
#include <unistd.h>

#include "esp_err.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "nvs_flash.h"

#include "audio_capture.h"
#include "ckks_encrypt.h"
#include "net_client.h"

static const char *TAG = "exp10";  // for logging

#define AUDIO_FRAME_SAMPLES 1024

void app_main(void)
{
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    ESP_ERROR_CHECK(net_wifi_init_sta());
    ESP_ERROR_CHECK(audio_capture_init());

    static int16_t pcm[AUDIO_FRAME_SAMPLES];
    int sock = -1;

    while (1) {
        if (sock < 0) {
            sock = net_tcp_connect();
            if (sock < 0) {
                ESP_LOGW(TAG, "Server connect failed, retrying in 2s");
                vTaskDelay(pdMS_TO_TICKS(2000));
                continue;
            }
        }

        size_t got = 0;
        esp_err_t err = audio_capture_read(pcm, AUDIO_FRAME_SAMPLES, &got);
        if (err != ESP_OK || got == 0) {
            ESP_LOGW(TAG, "Audio read failed: %s", esp_err_to_name(err));
            continue;
        }

        uint8_t *payload = NULL;
        size_t payload_len = 0;
        ESP_ERROR_CHECK(ckks_encrypt_frame(pcm, got, &payload, &payload_len));

        if (net_tcp_send_all(sock, payload, payload_len) != ESP_OK) {
            close(sock);
            sock = -1;
        }
        free(payload);
    }
}