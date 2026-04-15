#include "udp_sender.h"

#include <errno.h>
#include <string.h>
#include <unistd.h>

#include "esp_log.h"
#include "lwip/inet.h"
#include "lwip/sockets.h"

static const char *TAG = "udp_sender";

static int s_sock = -1;
static struct sockaddr_in s_dest_addr;

int udp_sender_init(const char *ip, uint16_t port)
{
    s_sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (s_sock < 0) {
        ESP_LOGE(TAG, "socket() failed: errno=%d", errno);
        return -1;
    }

    memset(&s_dest_addr, 0, sizeof(s_dest_addr));
    s_dest_addr.sin_family = AF_INET;
    s_dest_addr.sin_port = htons(port);

    if (inet_pton(AF_INET, ip, &s_dest_addr.sin_addr) != 1) {
        ESP_LOGE(TAG, "Invalid target IP: %s", ip);
        close(s_sock);
        s_sock = -1;
        return -1;
    }

    ESP_LOGI(TAG, "UDP target set to %s:%u", ip, (unsigned)port);
    return 0;
}

int udp_sender_send(const uint8_t *data, size_t len)
{
    if (s_sock < 0) {
        return -1;
    }

    int sent = sendto(s_sock, data, len, 0, (struct sockaddr *)&s_dest_addr, sizeof(s_dest_addr));
    if (sent < 0) {
        ESP_LOGW(TAG, "sendto() failed: errno=%d", errno);
        return -1;
    }

    return sent;
}

void udp_sender_deinit(void)
{
    if (s_sock >= 0) {
        close(s_sock);
        s_sock = -1;
    }
}
