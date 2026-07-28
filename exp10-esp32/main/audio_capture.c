#include "audio_capture.h"

#include "driver/i2s_std.h"
#include "esp_log.h"
#include "sdkconfig.h"

static const char *TAG = "audio_capture";
static i2s_chan_handle_t rx_chan = NULL;

esp_err_t audio_capture_init(void)
{
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_AUTO, I2S_ROLE_MASTER);
    ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, NULL, &rx_chan));

    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(CONFIG_HDSP_SAMPLE_RATE_HZ),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT,
                                                         I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = CONFIG_HDSP_I2S_SCK_GPIO,
            .ws = CONFIG_HDSP_I2S_WS_GPIO,
            .dout = I2S_GPIO_UNUSED,
            .din = CONFIG_HDSP_I2S_SD_GPIO,
            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv = false,
            },
        },
    };
    /* INMP441 L/R pin tied to GND drives data on the left slot. */
    std_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;

    ESP_ERROR_CHECK(i2s_channel_init_std_mode(rx_chan, &std_cfg));
    ESP_ERROR_CHECK(i2s_channel_enable(rx_chan));

    ESP_LOGI(TAG, "I2S RX channel ready at %d Hz (ws=%d sck=%d sd=%d)",
              CONFIG_HDSP_SAMPLE_RATE_HZ, CONFIG_HDSP_I2S_WS_GPIO,
              CONFIG_HDSP_I2S_SCK_GPIO, CONFIG_HDSP_I2S_SD_GPIO);
    return ESP_OK;
}

esp_err_t audio_capture_read(int16_t *out, size_t max_samples, size_t *out_count)
{
    size_t bytes_read = 0;
    esp_err_t err = i2s_channel_read(rx_chan, out, max_samples * sizeof(int16_t), &bytes_read, portMAX_DELAY);
    if (err != ESP_OK) {
        *out_count = 0;
        return err;
    }
    *out_count = bytes_read / sizeof(int16_t);
    return ESP_OK;
}

void audio_capture_deinit(void)
{
    if (rx_chan) {
        i2s_channel_disable(rx_chan);
        i2s_del_channel(rx_chan);
        rx_chan = NULL;
    }
}