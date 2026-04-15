#ifndef UDP_SENDER_H
#define UDP_SENDER_H

#include <stddef.h>
#include <stdint.h>

int udp_sender_init(const char *ip, uint16_t port);
int udp_sender_send(const uint8_t *data, size_t len);
void udp_sender_deinit(void);

#endif

