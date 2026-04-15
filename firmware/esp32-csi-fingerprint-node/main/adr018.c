#include "adr018.h"

#include <string.h>

static uint32_t channel_to_mhz(uint8_t channel)
{
    if (channel >= 1 && channel <= 13) {
        return 2412 + (uint32_t)(channel - 1) * 5;
    }
    if (channel == 14) {
        return 2484;
    }
    if (channel >= 36 && channel <= 177) {
        return 5000 + (uint32_t)channel * 5;
    }
    return 0;
}

size_t adr018_serialize_frame(const wifi_csi_info_t *info,
                              uint8_t node_id,
                              uint32_t sequence,
                              uint8_t *buf,
                              size_t buf_len)
{
    if (info == NULL || buf == NULL || info->buf == NULL) {
        return 0;
    }

    uint16_t iq_len = (uint16_t)info->len;
    size_t frame_size = ADR018_HEADER_SIZE + iq_len;
    if (frame_size > buf_len) {
        return 0;
    }

    uint16_t n_subcarriers = (uint16_t)(iq_len / 2);
    uint8_t n_antennas = 1;
    uint32_t freq_mhz = channel_to_mhz(info->rx_ctrl.channel);

    uint32_t magic = ADR018_MAGIC;
    memcpy(&buf[0], &magic, sizeof(magic));
    buf[4] = node_id;
    buf[5] = n_antennas;
    memcpy(&buf[6], &n_subcarriers, sizeof(n_subcarriers));
    memcpy(&buf[8], &freq_mhz, sizeof(freq_mhz));
    memcpy(&buf[12], &sequence, sizeof(sequence));
    buf[16] = (uint8_t)(int8_t)info->rx_ctrl.rssi;
    buf[17] = (uint8_t)(int8_t)info->rx_ctrl.noise_floor;
    buf[18] = 0;
    buf[19] = 0;
    memcpy(&buf[ADR018_HEADER_SIZE], info->buf, iq_len);

    return frame_size;
}

