idf_component_register(
    SRCS
        "main.c"
        "app_config.c"
        "wifi_station.c"
        "udp_sender.c"
        "adr018.c"
        "csi_fingerprint.c"
    INCLUDE_DIRS "."
    REQUIRES esp_event esp_netif esp_timer esp_wifi mdns nvs_flash
)
