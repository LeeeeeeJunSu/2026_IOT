#include "csi_fingerprint.h"

#include <string.h>

#include "adr018.h"
#include "esp_err.h"
#include "esp_log.h"
#include "esp_wifi.h"
#include "udp_sender.h"

static const char *TAG = "csi_fingerprint";

static uint32_t s_sequence = 0;
static uint8_t s_node_id = 0;
static uint32_t s_cb_count = 0;
static uint32_t s_send_failures = 0;

static void wifi_promiscuous_cb(void *buf, wifi_promiscuous_pkt_type_t type)
{
    (void)buf;
    (void)type;
}

static void wifi_csi_cb(void *ctx, wifi_csi_info_t *info)
{
    (void)ctx;
    s_cb_count++;

    uint8_t frame[ADR018_MAX_FRAME_SIZE];
    size_t frame_len = adr018_serialize_frame(info, s_node_id, s_sequence++, frame, sizeof(frame));
    if (frame_len == 0) {
        return;
    }

    int sent = udp_sender_send(frame, frame_len);
    if (sent < 0) {
        s_send_failures++;
        if (s_send_failures <= 5) {
            ESP_LOGW(TAG, "UDP send failed (%lu)", (unsigned long)s_send_failures);
        }
        return;
    }

    if (s_cb_count <= 3 || (s_cb_count % 200) == 0) {
        ESP_LOGI(TAG, "CSI frame #%lu len=%d rssi=%d ch=%d",
                 (unsigned long)s_cb_count,
                 info->len,
                 info->rx_ctrl.rssi,
                 info->rx_ctrl.channel);
    }
}

esp_err_t csi_fingerprint_start(uint8_t node_id)
{
    s_node_id = node_id;
    s_sequence = 0;
    s_cb_count = 0;
    s_send_failures = 0;

    ESP_ERROR_CHECK(esp_wifi_set_promiscuous(true));
    ESP_ERROR_CHECK(esp_wifi_set_promiscuous_rx_cb(wifi_promiscuous_cb));

    wifi_promiscuous_filter_t filter = {
        .filter_mask = WIFI_PROMIS_FILTER_MASK_MGMT | WIFI_PROMIS_FILTER_MASK_DATA,
    };
    ESP_ERROR_CHECK(esp_wifi_set_promiscuous_filter(&filter));

    wifi_csi_config_t csi_cfg = {
        .lltf_en = true,
        .htltf_en = true,
        .stbc_htltf2_en = true,
        .ltf_merge_en = true,
        .channel_filter_en = false,
        .manu_scale = false,
        .shift = false,
    };

    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi_cfg));
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(wifi_csi_cb, NULL));
    ESP_ERROR_CHECK(esp_wifi_set_csi(true));

    ESP_LOGI(TAG, "CSI fingerprint capture enabled for node %u", (unsigned)s_node_id);
    return ESP_OK;
}

void csi_fingerprint_stop(void)
{
    esp_wifi_set_csi(false);
    esp_wifi_set_promiscuous(false);
}
