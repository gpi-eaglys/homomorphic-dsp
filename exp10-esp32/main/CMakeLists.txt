idf_component_register(
    SRCS "main.c" "audio_capture.c" "net_client.c" "ckks_encrypt.c"
    INCLUDE_DIRS "."
    REQUIRES driver esp_driver_i2s esp_wifi esp_netif esp_event nvs_flash lwip
)