#ifndef CSI_FINGERPRINT_H
#define CSI_FINGERPRINT_H

#include <stdint.h>

#include "esp_err.h"

esp_err_t csi_fingerprint_start(uint8_t node_id, uint16_t send_interval_ms);
void csi_fingerprint_stop(void);

#endif
