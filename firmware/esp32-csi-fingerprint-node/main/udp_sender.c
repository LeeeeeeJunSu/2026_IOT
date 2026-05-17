#include "udp_sender.h"

#include <errno.h>
#include <stdbool.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>

#include "esp_err.h"
#include "esp_log.h"
#include "lwip/inet.h"
#include "lwip/sockets.h"
#include "mdns.h"

static const char *TAG = "udp_sender";

static int s_sock = -1;
static struct sockaddr_in s_dest_addr;

static bool strip_local_suffix(const char *host, char *out, size_t out_len)
{
    const char suffix[] = ".local";
    size_t host_len;
    size_t suffix_len = sizeof(suffix) - 1;

    if (host == NULL || out == NULL || out_len == 0) {
        return false;
    }

    host_len = strlen(host);
    if (host_len > suffix_len && strcasecmp(host + host_len - suffix_len, suffix) == 0) {
        size_t copy_len = host_len - suffix_len;
        if (copy_len >= out_len) {
            copy_len = out_len - 1;
        }
        memcpy(out, host, copy_len);
        out[copy_len] = '\0';
        return true;
    }

    snprintf(out, out_len, "%s", host);
    return false;
}

static int resolve_mdns_ipv4(const char *host, struct in_addr *addr)
{
    char mdns_host[64];
    esp_ip4_addr_t resolved_addr;
    esp_err_t err;

    strip_local_suffix(host, mdns_host, sizeof(mdns_host));
    ESP_LOGI(TAG, "Resolving mDNS host %s.local", mdns_host);

    err = mdns_init();
    if (err != ESP_OK && err != ESP_ERR_INVALID_STATE) {
        ESP_LOGE(TAG, "mdns_init() failed: %s", esp_err_to_name(err));
        return -1;
    }

    err = mdns_query_a(mdns_host, 5000, &resolved_addr);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "mDNS query failed for %s.local: %s", mdns_host, esp_err_to_name(err));
        return -1;
    }

    addr->s_addr = resolved_addr.addr;
    ESP_LOGI(TAG, "mDNS resolved %s.local to " IPSTR, mdns_host, IP2STR(&resolved_addr));
    return 0;
}

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
        if (resolve_mdns_ipv4(ip, &s_dest_addr.sin_addr) != 0) {
            ESP_LOGE(TAG, "Invalid or unresolved UDP target: %s", ip);
            close(s_sock);
            s_sock = -1;
            return -1;
        }
    }

    char resolved_ip[INET_ADDRSTRLEN] = { 0 };
    inet_ntop(AF_INET, &s_dest_addr.sin_addr, resolved_ip, sizeof(resolved_ip));
    ESP_LOGI(TAG, "UDP target set to %s:%u", resolved_ip, (unsigned)port);
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
