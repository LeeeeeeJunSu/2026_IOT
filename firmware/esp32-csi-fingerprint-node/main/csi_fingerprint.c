#include "csi_fingerprint.h"

#include <string.h>

#include "adr018.h"
#include "esp_err.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"
#include "udp_sender.h"

static const char *TAG = "csi_fingerprint";

static uint32_t s_sequence = 0;
static uint8_t s_node_id = 0;
static uint32_t s_cb_count = 0;
static uint32_t s_skipped_count = 0;
static uint32_t s_send_failures = 0;
static uint32_t s_queue_drops = 0;
static int64_t s_last_send_us = 0;
static uint32_t s_send_interval_us = 20000;
static QueueHandle_t s_send_queue = NULL;
static TaskHandle_t s_send_task = NULL;

typedef struct {
    size_t len;
    uint8_t data[ADR018_MAX_FRAME_SIZE];
} csi_udp_frame_t;

static void udp_send_task(void *arg)
{
    (void)arg;
    csi_udp_frame_t item;
    while (1) {
        if (xQueueReceive(s_send_queue, &item, portMAX_DELAY) != pdTRUE) {
            continue;
        }
        int sent = udp_sender_send(item.data, item.len);
        if (sent < 0) {
            s_send_failures++;
            if (s_send_failures <= 5 || (s_send_failures % 1000) == 0) {
                ESP_LOGW(TAG, "UDP send failed (%lu)", (unsigned long)s_send_failures);
            }
        }
    }
}

static void wifi_promiscuous_cb(void *buf, wifi_promiscuous_pkt_type_t type)
{
    (void)buf;
    (void)type;
}

static void wifi_csi_cb(void *ctx, wifi_csi_info_t *info)
{
    (void)ctx;
    s_cb_count++;

    int64_t now_us = esp_timer_get_time();
    if (s_send_interval_us > 0 && s_last_send_us > 0 &&
        now_us - s_last_send_us < (int64_t)s_send_interval_us) {
        s_skipped_count++;
        return;
    }

    uint8_t frame[ADR018_MAX_FRAME_SIZE];
    size_t frame_len = adr018_serialize_frame(info, s_node_id, s_sequence++, frame, sizeof(frame));
    if (frame_len == 0) {
        return;
    }

    csi_udp_frame_t item = {
        .len = frame_len,
    };
    memcpy(item.data, frame, frame_len);
    if (xQueueSend(s_send_queue, &item, 0) != pdTRUE) {
        s_queue_drops++;
        if (s_queue_drops <= 5 || (s_queue_drops % 1000) == 0) {
            ESP_LOGW(TAG, "UDP send queue full, dropped=%lu", (unsigned long)s_queue_drops);
        }
        return;
    }
    s_last_send_us = now_us;

    if (s_cb_count <= 3 || (s_cb_count % 200) == 0) {
        ESP_LOGI(TAG, "CSI frame #%lu queued_seq=%lu skipped=%lu dropped=%lu len=%d rssi=%d ch=%d",
                 (unsigned long)s_cb_count,
                 (unsigned long)s_sequence,
                 (unsigned long)s_skipped_count,
                 (unsigned long)s_queue_drops,
                 info->len,
                 info->rx_ctrl.rssi,
                 info->rx_ctrl.channel);
    }
}

esp_err_t csi_fingerprint_start(uint8_t node_id, uint16_t send_interval_ms)
{
    s_node_id = node_id;
    s_sequence = 0;
    s_cb_count = 0;
    s_skipped_count = 0;
    s_send_failures = 0;
    s_queue_drops = 0;
    s_last_send_us = 0;
    s_send_interval_us = (uint32_t)send_interval_ms * 1000U;

    if (s_send_queue == NULL) {
        s_send_queue = xQueueCreate(32, sizeof(csi_udp_frame_t));
        if (s_send_queue == NULL) {
            ESP_LOGE(TAG, "Failed to create UDP send queue");
            return ESP_ERR_NO_MEM;
        }
    } else {
        xQueueReset(s_send_queue);
    }

    if (s_send_task == NULL) {
        BaseType_t created = xTaskCreatePinnedToCore(
            udp_send_task,
            "csi_udp_send",
            4096,
            NULL,
            18,
            &s_send_task,
            1);
        if (created != pdPASS) {
            ESP_LOGE(TAG, "Failed to create UDP send task");
            return ESP_ERR_NO_MEM;
        }
    }

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

    ESP_LOGI(TAG,
             "CSI fingerprint capture enabled for node %u, send_interval_ms=%u",
             (unsigned)s_node_id,
             (unsigned)send_interval_ms);
    return ESP_OK;
}

void csi_fingerprint_stop(void)
{
    esp_wifi_set_csi(false);
    esp_wifi_set_promiscuous(false);
    if (s_send_queue != NULL) {
        xQueueReset(s_send_queue);
    }
}
