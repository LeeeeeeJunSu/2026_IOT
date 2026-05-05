#include "app_config.h"
#include "csi_fingerprint.h"
#include "esp_err.h"
#include "esp_log.h"
#include "esp_system.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "nvs_flash.h"
#include "udp_sender.h"
#include "wifi_station.h"

static const char *TAG = "main";

static void init_nvs(void)
{
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);
}

void app_main(void)
{
    app_config_t cfg = { 0 };

    init_nvs();
    app_config_load(&cfg);

    ESP_LOGI(TAG, "ESP32 CSI fingerprint node ready");
    ESP_LOGI(TAG, "node_id=%u target=%s:%u ssid=%s channel=%u send_interval_ms=%u",
             (unsigned)cfg.node_id,
             cfg.target_ip,
             (unsigned)cfg.target_port,
             cfg.wifi_ssid,
             (unsigned)cfg.wifi_channel,
             (unsigned)cfg.csi_send_interval_ms);

    csi_wifi_station_start(&cfg);

    if (udp_sender_init(cfg.target_ip, cfg.target_port) != 0) {
        ESP_LOGE(TAG, "Failed to open UDP sender");
        return;
    }

    ESP_ERROR_CHECK(csi_fingerprint_start(cfg.node_id, cfg.csi_send_interval_ms));

    while (1) {
        vTaskDelay(pdMS_TO_TICKS(10000));
    }
}
