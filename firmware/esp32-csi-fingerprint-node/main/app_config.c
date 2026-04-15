#include "app_config.h"

#include <stdio.h>
#include <string.h>

#include "esp_log.h"
#include "nvs.h"
#include "nvs_flash.h"
#include "sdkconfig.h"

static const char *TAG = "app_config";

static void copy_default(char *dst, size_t dst_len, const char *src)
{
    if (dst_len == 0) {
        return;
    }
    if (src == NULL) {
        dst[0] = '\0';
        return;
    }
    snprintf(dst, dst_len, "%s", src);
}

void app_config_load(app_config_t *cfg)
{
    if (cfg == NULL) {
        return;
    }

    copy_default(cfg->wifi_ssid, sizeof(cfg->wifi_ssid), CONFIG_CSI_DEFAULT_WIFI_SSID);
    copy_default(cfg->wifi_password, sizeof(cfg->wifi_password), CONFIG_CSI_DEFAULT_WIFI_PASSWORD);
    copy_default(cfg->target_ip, sizeof(cfg->target_ip), CONFIG_CSI_DEFAULT_TARGET_IP);
    cfg->target_port = (uint16_t)CONFIG_CSI_DEFAULT_TARGET_PORT;
    cfg->node_id = (uint8_t)CONFIG_CSI_DEFAULT_NODE_ID;
    cfg->wifi_channel = (uint8_t)CONFIG_CSI_DEFAULT_WIFI_CHANNEL;

    nvs_handle_t handle;
    esp_err_t err = nvs_open("csi_cfg", NVS_READONLY, &handle);
    if (err != ESP_OK) {
        ESP_LOGI(TAG, "No NVS provisioning found, using defaults");
        return;
    }

    size_t len;
    char tmp[APP_CFG_PASS_MAX];

    len = sizeof(tmp);
    if (nvs_get_str(handle, "ssid", tmp, &len) == ESP_OK && len > 1) {
        copy_default(cfg->wifi_ssid, sizeof(cfg->wifi_ssid), tmp);
        ESP_LOGI(TAG, "NVS override: ssid=%s", cfg->wifi_ssid);
    }

    len = sizeof(tmp);
    if (nvs_get_str(handle, "password", tmp, &len) == ESP_OK) {
        copy_default(cfg->wifi_password, sizeof(cfg->wifi_password), tmp);
        ESP_LOGI(TAG, "NVS override: password=***");
    }

    len = sizeof(tmp);
    if (nvs_get_str(handle, "target_ip", tmp, &len) == ESP_OK && len > 1) {
        copy_default(cfg->target_ip, sizeof(cfg->target_ip), tmp);
        ESP_LOGI(TAG, "NVS override: target_ip=%s", cfg->target_ip);
    }

    uint16_t port = 0;
    if (nvs_get_u16(handle, "target_port", &port) == ESP_OK) {
        cfg->target_port = port;
        ESP_LOGI(TAG, "NVS override: target_port=%u", cfg->target_port);
    }

    uint8_t node_id = 0;
    if (nvs_get_u8(handle, "node_id", &node_id) == ESP_OK) {
        cfg->node_id = node_id;
        ESP_LOGI(TAG, "NVS override: node_id=%u", cfg->node_id);
    }

    uint8_t channel = 0;
    if (nvs_get_u8(handle, "wifi_channel", &channel) == ESP_OK) {
        cfg->wifi_channel = channel;
        ESP_LOGI(TAG, "NVS override: wifi_channel=%u", cfg->wifi_channel);
    }

    nvs_close(handle);
}
