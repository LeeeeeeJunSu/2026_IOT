#ifndef ADR018_H
#define ADR018_H

#include <stddef.h>
#include <stdint.h>

#include "esp_wifi_types.h"

#define ADR018_MAGIC        0xC5110001
#define ADR018_HEADER_SIZE   20
#define ADR018_MAX_IQ_BYTES  1024
#define ADR018_MAX_FRAME_SIZE (ADR018_HEADER_SIZE + ADR018_MAX_IQ_BYTES)

size_t adr018_serialize_frame(const wifi_csi_info_t *info,
                              uint8_t node_id,
                              uint32_t sequence,
                              uint8_t *buf,
                              size_t buf_len);

#endif
