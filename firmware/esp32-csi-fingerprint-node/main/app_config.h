#ifndef APP_CONFIG_H
#define APP_CONFIG_H

#include <stdint.h>

#define APP_CFG_SSID_MAX   33
#define APP_CFG_PASS_MAX   65
#define APP_CFG_IP_MAX     16

typedef struct {
    char wifi_ssid[APP_CFG_SSID_MAX];
    char wifi_password[APP_CFG_PASS_MAX];
    char target_ip[APP_CFG_IP_MAX];
    uint16_t target_port;
    uint8_t node_id;
    uint8_t wifi_channel;
} app_config_t;

void app_config_load(app_config_t *cfg);

#endif

