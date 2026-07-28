#pragma once

#include <stddef.h>
#include <stdint.h>
#include "esp_err.h"

/* Connects to WiFi in station mode and blocks until an IP is obtained. */
esp_err_t net_wifi_init_sta(void);

/* Opens a TCP connection to CONFIG_HDSP_SERVER_HOST:CONFIG_HDSP_SERVER_PORT.
 * Returns a socket fd, or -1 on failure. */
int net_tcp_connect(void);

esp_err_t net_tcp_send_all(int sock, const uint8_t *data, size_t len);